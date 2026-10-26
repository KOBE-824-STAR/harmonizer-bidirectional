"""Authoritative nusc_pair_6cam input -> gt temporal dataset.

The current slot comes from the selected manifest group's edited ``input``.
The four history slots follow the official nuScenes sample chain and contain
the original RGB camera keyframes.  This is the teacher-forced analogue of
the official Harmonizer inference contract (current input followed by prior
predictions), and avoids inventing an asset trajectory by selecting arbitrary
asset-conditioned groups from older frames.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..constants import CAMERA_ORDER
from ..layout_temporal import OFFICIAL_TEMPORAL_OFFSETS


INDEX_SCHEMA = "scheme_a_tmv_nusc_pair_temporal_index_v2_calibrated"
TEMPORAL_POLICY = "current_group_input_then_four_prior_nuscenes_original_keyframes"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_from_filename(filename: str) -> str:
    parts = Path(filename).parts
    if len(parts) < 2 or parts[0] not in {"samples", "sweeps"}:
        raise ValueError(f"Not a nuScenes camera filename: {filename!r}")
    camera = parts[1]
    if camera not in CAMERA_ORDER:
        raise ValueError(f"Unexpected camera in nuScenes metadata: {camera!r}")
    return camera


def _canonical_views(row: dict[str, Any]) -> list[dict[str, Any]]:
    views = row.get("views")
    if not isinstance(views, list) or len(views) != len(CAMERA_ORDER):
        raise ValueError(f"Group {row.get('group_id')} must contain V=6 views")
    by_name: dict[str, dict[str, Any]] = {}
    for view in views:
        name = view.get("camera_name")
        if name not in CAMERA_ORDER:
            raise ValueError(f"Unexpected camera {name!r} in {row.get('group_id')}")
        if name in by_name:
            raise ValueError(f"Duplicate camera {name!r} in {row.get('group_id')}")
        by_name[name] = view
    missing = set(CAMERA_ORDER) - set(by_name)
    if missing:
        raise ValueError(f"Missing cameras in {row.get('group_id')}: {sorted(missing)}")
    return [by_name[name] for name in CAMERA_ORDER]


def _published_role_path(dataset_root: Path, view: dict[str, Any], role: str) -> str:
    sources = view.get("role_sources")
    if not isinstance(sources, dict) or role not in sources:
        raise ValueError(f"Camera {view.get('camera_name')} is missing role {role!r}")
    relative = sources[role].get("published_relative_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Camera {view.get('camera_name')} has no published {role} path")
    path = (dataset_root / relative).resolve()
    try:
        path.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"Published path escapes dataset root: {relative}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _atomic_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=5)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _quaternion_wxyz_to_rotation(value: Any) -> list[list[float]]:
    """Convert a nuScenes unit quaternion ``[w,x,y,z]`` to a rotation matrix."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,) or not np.isfinite(array).all():
        raise ValueError("calibrated_sensor.rotation must be four finite values")
    norm = float(np.linalg.norm(array))
    if norm <= 0:
        raise ValueError("calibrated_sensor.rotation quaternion has zero norm")
    w, x, y, z = array / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    ).tolist()


def _resized_camera_calibration(
    sample_data: dict[str, Any],
    calibrated_sensor: dict[str, Any],
    *,
    height: int = 288,
    width: int = 512,
) -> tuple[list[list[float]], list[list[float]]]:
    """Return resized K and nuScenes sensor-to-ego transform for one camera."""

    original_width = int(sample_data.get("width", 0))
    original_height = int(sample_data.get("height", 0))
    if original_width < 1 or original_height < 1:
        raise ValueError("camera sample_data must include positive width/height")
    intrinsic = np.asarray(calibrated_sensor.get("camera_intrinsic"), dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("camera_intrinsic must be a finite 3x3 matrix")
    intrinsic = intrinsic.copy()
    intrinsic[0] *= float(width) / float(original_width)
    intrinsic[1] *= float(height) / float(original_height)

    translation = np.asarray(calibrated_sensor.get("translation"), dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("calibrated_sensor.translation must be three finite values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        _quaternion_wxyz_to_rotation(calibrated_sensor.get("rotation")),
        dtype=np.float64,
    )
    transform[:3, 3] = translation
    return intrinsic.tolist(), transform.tolist()


def build_temporal_index_cache(
    dataset_root: str | Path,
    split: str,
    cache_path: str | Path,
    *,
    nuscenes_version_root: str | Path = "/mnt/public_data/nuScenes/v1.0-trainval",
    temporal_stride: int = 1,
) -> dict[str, Any]:
    """Build the compact, provenance-bearing real-history index."""

    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    if temporal_stride != 1:
        raise ValueError("The formal Scheme A baseline requires temporal_stride=1")
    root = Path(dataset_root).expanduser().resolve()
    version_root = Path(nuscenes_version_root).expanduser().resolve()
    manifest = root / "metadata" / f"{split}_groups.jsonl"
    sample_json = version_root / "sample.json"
    sample_data_json = version_root / "sample_data.json"
    calibrated_sensor_json = version_root / "calibrated_sensor.json"
    for path in (manifest, sample_json, sample_data_json, calibrated_sensor_json):
        if not path.is_file():
            raise FileNotFoundError(path)

    samples = {row["token"]: row for row in json.loads(sample_json.read_text())}
    keyframes: dict[tuple[str, str], dict[str, Any]] = {}
    for row in json.loads(sample_data_json.read_text()):
        if row.get("is_key_frame") and str(row.get("filename", "")).startswith("samples/CAM_"):
            key = (row["sample_token"], _camera_from_filename(row["filename"]))
            if key in keyframes:
                raise ValueError(f"Duplicate official keyframe metadata: {key}")
            keyframes[key] = row
    calibrated_sensors = {
        row["token"]: row for row in json.loads(calibrated_sensor_json.read_text())
    }

    records: list[dict[str, Any]] = []
    discarded_insufficient_history = 0
    with manifest.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != 2 or row.get("split") != split:
                raise ValueError(f"Unexpected manifest schema/split at line {line_number}")
            if row.get("review", {}).get("status") != "accepted":
                raise ValueError(f"Non-accepted authoritative group at line {line_number}")
            current_sample = samples.get(row.get("sample_token"))
            if current_sample is None:
                raise ValueError(f"Unknown sample_token at line {line_number}")
            prior_samples: list[dict[str, Any]] = []
            cursor = current_sample
            for _ in OFFICIAL_TEMPORAL_OFFSETS[1:]:
                previous_token = cursor.get("prev", "")
                if not previous_token:
                    break
                cursor = samples.get(previous_token)
                if cursor is None:
                    raise ValueError(f"Broken nuScenes sample.prev chain at line {line_number}")
                if cursor.get("scene_token") != current_sample.get("scene_token"):
                    raise ValueError(f"Temporal chain crosses scene at line {line_number}")
                prior_samples.append(cursor)
            if len(prior_samples) != 4:
                discarded_insufficient_history += 1
                continue

            views = _canonical_views(row)
            for view in views:
                exposure = view.get("exposure", {})
                if exposure.get("sample_token") != row["sample_token"]:
                    raise ValueError(f"Exposure/sample mismatch at line {line_number}")
                if exposure.get("camera_name") != view["camera_name"]:
                    raise ValueError(f"Exposure/camera mismatch at line {line_number}")
            histories: list[list[str]] = []
            history_tokens: list[str] = []
            for prior in prior_samples:
                paths: list[str] = []
                history_tokens.append(prior["token"])
                for camera in CAMERA_ORDER:
                    metadata = keyframes.get((prior["token"], camera))
                    if metadata is None:
                        raise ValueError(
                            f"Missing {camera} keyframe for prior sample {prior['token']}"
                        )
                    image_path = version_root.parent / metadata["filename"]
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                    paths.append(str(image_path.resolve()))
                histories.append(paths)

            current_roles = {
                role: [_published_role_path(root, view, role) for view in views]
                for role in ("input", "gt", "target")
            }
            intrinsics: list[list[list[float]]] = []
            T_ego_from_cam: list[list[list[float]]] = []
            calibrated_sensor_tokens: list[str] = []
            for camera in CAMERA_ORDER:
                metadata = keyframes.get((row["sample_token"], camera))
                if metadata is None:
                    raise ValueError(
                        f"Missing current {camera} keyframe metadata at line {line_number}"
                    )
                token = metadata.get("calibrated_sensor_token")
                calibration = calibrated_sensors.get(token)
                if calibration is None:
                    raise ValueError(
                        f"Missing calibrated_sensor {token!r} at line {line_number}"
                    )
                resized_k, sensor_to_ego = _resized_camera_calibration(
                    metadata, calibration
                )
                intrinsics.append(resized_k)
                T_ego_from_cam.append(sensor_to_ego)
                calibrated_sensor_tokens.append(str(token))
            records.append(
                {
                    "group_id": row["group_id"],
                    "context_id": row["context_id"],
                    "scene_name": row["scene_name"],
                    "current_frame_index": int(row["frame_index"]),
                    "sample_token": row["sample_token"],
                    "history_sample_tokens": history_tokens,
                    "temporal_frame_indices": [
                        int(row["frame_index"]) + offset
                        for offset in OFFICIAL_TEMPORAL_OFFSETS
                    ],
                    "input_paths": current_roles["input"],
                    "gt_paths": current_roles["gt"],
                    "target_paths": current_roles["target"],
                    "history_paths": histories,
                    "intrinsics": intrinsics,
                    "T_ego_from_cam": T_ego_from_cam,
                    "calibrated_sensor_tokens": calibrated_sensor_tokens,
                }
            )

    provenance = {
        "schema": INDEX_SCHEMA,
        "dataset_root": str(root),
        "split": split,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "nuscenes_version_root": str(version_root),
        "sample_json_sha256": sha256_file(sample_json),
        "sample_data_json_sha256": sha256_file(sample_data_json),
        "calibrated_sensor_json_sha256": sha256_file(calibrated_sensor_json),
        "T": len(OFFICIAL_TEMPORAL_OFFSETS),
        "V": len(CAMERA_ORDER),
        "camera_order": list(CAMERA_ORDER),
        "rgb_slot_offsets": list(OFFICIAL_TEMPORAL_OFFSETS),
        "temporal_stride": temporal_stride,
        "construction_policy": TEMPORAL_POLICY,
        "history_role": "official_nuscenes_original_rgb_keyframe",
        "current_input_role": "input",
        "current_supervision_role": "gt",
        "target_in_main_loss": False,
        "sample_count": len(records),
        "discarded_insufficient_history": discarded_insufficient_history,
    }
    payload = {"provenance": provenance, "records": records}
    _atomic_pickle(Path(cache_path), payload)
    return provenance


def _load_rgb(path: str, height: int, width: int) -> torch.Tensor:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        if image.size != (width, height):
            resampling = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
            image = image.resize((width, height), resampling)
        array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)


class NuscPairTemporalDataset(Dataset[dict[str, Any]]):
    """Return real ``[T=5,V=6,C,H,W]`` input and current-frame supervision."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        height: int = 288,
        width: int = 512,
        load_target_images: bool = False,
    ) -> None:
        self.cache_path = Path(cache_path).expanduser().resolve()
        with self.cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        provenance = payload.get("provenance", {})
        if provenance.get("schema") != INDEX_SCHEMA:
            raise ValueError(f"Unsupported temporal cache: {self.cache_path}")
        if tuple(provenance.get("camera_order", ())) != CAMERA_ORDER:
            raise ValueError("Temporal cache camera ordering is incompatible")
        if tuple(provenance.get("rgb_slot_offsets", ())) != OFFICIAL_TEMPORAL_OFFSETS:
            raise ValueError("Temporal cache does not use official T=5 ordering")
        self.records: list[dict[str, Any]] = payload["records"]
        self.provenance: dict[str, Any] = provenance
        self.height = int(height)
        self.width = int(width)
        self.load_target_images = bool(load_target_images)
        if self.height != 288 or self.width != 512:
            raise ValueError("Formal nusc_pair temporal dataset is fixed at 288x512")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        current = torch.stack(
            [_load_rgb(path, self.height, self.width) for path in record["input_paths"]]
        )
        history = [
            torch.stack([_load_rgb(path, self.height, self.width) for path in paths])
            for paths in record["history_paths"]
        ]
        input_images = torch.stack([current, *history])
        gt_images = torch.stack(
            [_load_rgb(path, self.height, self.width) for path in record["gt_paths"]]
        )
        result: dict[str, Any] = {
            "input_images": input_images,
            "gt_images": gt_images,
            "view_valid": torch.ones((5, 6), dtype=torch.bool),
            "camera_ids": torch.arange(6, dtype=torch.long),
            "temporal_offsets": torch.tensor(OFFICIAL_TEMPORAL_OFFSETS, dtype=torch.long),
            "temporal_frame_indices": torch.tensor(
                record["temporal_frame_indices"], dtype=torch.long
            ),
            "intrinsics": torch.tensor(record["intrinsics"], dtype=torch.float32),
            "T_ego_from_cam": torch.tensor(
                record["T_ego_from_cam"], dtype=torch.float32
            ),
            "group_id": record["group_id"],
            "context_id": record["context_id"],
            "scene": record["scene_name"],
            "current_frame_index": record["current_frame_index"],
            "sample_token": record["sample_token"],
            "history_sample_tokens": tuple(record["history_sample_tokens"]),
            "camera_names": CAMERA_ORDER,
            "target_paths": tuple(record["target_paths"]),
        }
        if self.load_target_images:
            result["target_images"] = torch.stack(
                [_load_rgb(path, self.height, self.width) for path in record["target_paths"]]
            )
        return result


def collate_temporal_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    tensor_keys = {
        "input_images",
        "gt_images",
        "view_valid",
        "camera_ids",
        "temporal_offsets",
        "temporal_frame_indices",
        "intrinsics",
        "T_ego_from_cam",
    }
    if "target_images" in samples[0]:
        tensor_keys.add("target_images")
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys}
    for key in samples[0]:
        if key not in tensor_keys:
            batch[key] = [sample[key] for sample in samples]
    return batch


def build_scheme_a_dataloader(
    dataset: NuscPairTemporalDataset,
    *,
    batch_size: int,
    sampler: Iterable[int] | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    if batch_size != 1:
        raise ValueError("Formal Scheme A uses per-GPU batch size 1")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_temporal_samples,
    )
