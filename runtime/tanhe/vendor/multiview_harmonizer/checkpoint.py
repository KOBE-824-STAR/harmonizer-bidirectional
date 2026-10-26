"""Strict loading and auditing for the official DiffusionHarmonizer checkpoint."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch

from .tokenizer_utils import inner_tokenizer_module


@dataclass(frozen=True)
class LoadAudit:
    official_parameter_tensors: int
    official_parameter_numel: int
    matched_parameter_tensors: int
    matched_parameter_numel: int
    missing_core: tuple[str, ...]
    shape_mismatch: tuple[str, ...]
    ignored_ema_tensors: int
    retained_runtime_state: tuple[str, ...]
    load_missing: tuple[str, ...]
    load_unexpected: tuple[str, ...]

    @property
    def tensor_coverage(self) -> float:
        return self.matched_parameter_tensors / max(self.official_parameter_tensors, 1)

    @property
    def numel_coverage(self) -> float:
        return self.matched_parameter_numel / max(self.official_parameter_numel, 1)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tensor_coverage"] = self.tensor_coverage
        payload["numel_coverage"] = self.numel_coverage
        return payload


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: str | Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")


def remap_official_unet_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    schema_families: set[str] = set()
    for key, value in state_dict.items():
        if key.startswith("net_ema."):
            mapped_key = f"dit_ema.{key[len('net_ema.'):]}"
            schema_families.add("legacy_net")
        elif key.startswith("net."):
            mapped_key = f"dit.{key[len('net.'):]}"
            schema_families.add("legacy_net")
        elif key.startswith("dit_ema.") or key.startswith("dit."):
            mapped_key = key
            schema_families.add("current_dit")
        else:
            raise ValueError(f"Unsupported official UNet checkpoint key: {key!r}")
        if mapped_key in remapped:
            raise RuntimeError(
                f"Official UNet key remap collision at {mapped_key!r}"
            )
        remapped[mapped_key] = value
    if len(schema_families) > 1:
        raise RuntimeError(
            "Official UNet checkpoint mixes legacy net.* and current dit.* schemas"
        )
    return remapped


def _canonical_runtime_dit_parameters(
    dit: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    """Return canonical state-dict names through a strict Parameter identity bijection.

    PyTorch activation-checkpoint wrappers intentionally expose an implementation
    segment through ``named_parameters`` while their state-dict hooks preserve
    the wrapped module's canonical checkpoint names.  Parameter identity is the
    non-heuristic bridge between those namespaces.
    """

    raw_names_by_id: dict[int, list[str]] = defaultdict(list)
    raw_parameters_by_id: dict[int, torch.nn.Parameter] = {}
    for raw_name, parameter in dit.named_parameters(remove_duplicate=False):
        raw_names_by_id[id(parameter)].append(raw_name)
        raw_parameters_by_id[id(parameter)] = parameter

    shared_raw = {
        identity: names for identity, names in raw_names_by_id.items() if len(names) != 1
    }
    if shared_raw:
        example = next(iter(shared_raw.values()))
        raise RuntimeError(
            "Live DiT contains shared/aliased parameter names; canonical audit is not "
            f"one-to-one: {example}"
        )

    state_names_by_id: dict[int, list[str]] = defaultdict(list)
    for state_name, value in dit.state_dict(keep_vars=True).items():
        if isinstance(value, torch.nn.Parameter):
            state_names_by_id[id(value)].append(state_name)

    shared_state = {
        identity: names for identity, names in state_names_by_id.items() if len(names) != 1
    }
    if shared_state:
        example = next(iter(shared_state.values()))
        raise RuntimeError(
            "Live DiT state_dict aliases a parameter under multiple canonical names: "
            f"{example}"
        )
    raw_ids = set(raw_names_by_id)
    state_ids = set(state_names_by_id)
    if raw_ids != state_ids:
        raise RuntimeError(
            "Could not establish a complete Parameter identity bijection between "
            "named_parameters and state_dict: "
            f"missing_from_state={len(raw_ids - state_ids)}, "
            f"unknown_in_state={len(state_ids - raw_ids)}"
        )

    return {
        f"dit.{state_names_by_id[identity][0]}": raw_parameters_by_id[identity]
        for identity in raw_names_by_id
    }


def _split_live_and_ema(
    remapped_pipeline_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    unsupported = [
        key
        for key in remapped_pipeline_state
        if not key.startswith("dit.") and not key.startswith("dit_ema.")
    ]
    if unsupported:
        raise RuntimeError(f"Unsupported remapped UNet keys: {unsupported[:8]}")
    live = {
        key[len("dit.") :]: value
        for key, value in remapped_pipeline_state.items()
        if key.startswith("dit.")
    }
    ema = {
        key[len("dit_ema.") :]: value
        for key, value in remapped_pipeline_state.items()
        if key.startswith("dit_ema.")
    }
    return live, ema


def _require_ema_mirrors_live(
    live: Mapping[str, torch.Tensor], ema: Mapping[str, torch.Tensor]
) -> None:
    if not ema:
        return
    if set(live) != set(ema):
        raise RuntimeError(
            "Official EMA/live DiT key sets differ: "
            f"missing_ema={len(set(live) - set(ema))}, "
            f"extra_ema={len(set(ema) - set(live))}"
        )
    mismatched = [
        name
        for name in live
        if tuple(live[name].shape) != tuple(ema[name].shape)
        or live[name].dtype != ema[name].dtype
    ]
    if mismatched:
        raise RuntimeError(
            f"Official EMA/live DiT tensor metadata differs: {mismatched[:8]}"
        )


def audit_official_dit_parameters(
    dit: torch.nn.Module,
    remapped_pipeline_state: Mapping[str, torch.Tensor],
) -> LoadAudit:
    """Require every original live-DiT parameter to be present with exact shape.

    Call this before adding A/B modules.  That makes the target parameter set
    exactly the official 0.6B Text2Image DiT used by DiffusionHarmonizer.
    """

    target = _canonical_runtime_dit_parameters(dit)
    live, _ = _split_live_and_ema(remapped_pipeline_state)
    runtime_state_names = set(dit.state_dict())
    extra_live = set(live) - runtime_state_names
    if extra_live:
        raise RuntimeError(
            "Official live DiT contains keys absent from the runtime state: "
            f"{sorted(extra_live)[:8]}"
        )
    missing: list[str] = []
    mismatched: list[str] = []
    matched_tensors = 0
    matched_numel = 0
    for key, parameter in target.items():
        source = live.get(key[len("dit.") :])
        if source is None:
            missing.append(key)
        elif tuple(source.shape) != tuple(parameter.shape):
            mismatched.append(
                f"{key}: checkpoint={tuple(source.shape)} target={tuple(parameter.shape)}"
            )
        else:
            matched_tensors += 1
            matched_numel += parameter.numel()
    return LoadAudit(
        official_parameter_tensors=len(target),
        official_parameter_numel=sum(parameter.numel() for parameter in target.values()),
        matched_parameter_tensors=matched_tensors,
        matched_parameter_numel=matched_numel,
        missing_core=tuple(missing),
        shape_mismatch=tuple(mismatched),
        ignored_ema_tensors=sum(key.startswith("dit_ema.") for key in remapped_pipeline_state),
        retained_runtime_state=(),
        load_missing=(),
        load_unexpected=(),
    )


def load_official_live_dit_state(
    dit: torch.nn.Module,
    remapped_pipeline_state: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    """Strictly load official live tensors and retain named runtime-only state.

    All runtime Parameters must already have passed ``audit_official_dit_parameters``.
    Runtime-only entries may therefore only be buffers or module ``_extra_state``;
    they are copied into a complete state dict and loaded with ``strict=True``.
    """

    live, ema = _split_live_and_ema(remapped_pipeline_state)
    _require_ema_mirrors_live(live, ema)
    runtime_state = dit.state_dict()
    extra_live = set(live) - set(runtime_state)
    if extra_live:
        raise RuntimeError(
            "Official live DiT contains keys absent from the runtime state: "
            f"{sorted(extra_live)[:8]}"
        )
    canonical_parameters = _canonical_runtime_dit_parameters(dit)
    parameter_names = {name[len("dit.") :] for name in canonical_parameters}
    missing_parameters = parameter_names - set(live)
    if missing_parameters:
        raise RuntimeError(
            "Official live DiT is missing runtime Parameters: "
            f"{sorted(missing_parameters)[:8]}"
        )

    for name, value in live.items():
        target = runtime_state[name]
        if not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError(f"Official live DiT state {name!r} is not tensor-compatible")
        if tuple(value.shape) != tuple(target.shape):
            raise RuntimeError(
                f"Official live DiT state shape mismatch for {name}: "
                f"checkpoint={tuple(value.shape)} runtime={tuple(target.shape)}"
            )
        runtime_state[name] = value

    retained = tuple(f"dit.{name}" for name in sorted(set(runtime_state) - set(live)))
    incompatible = dit.load_state_dict(runtime_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict live DiT load returned incompatible keys: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    return retained


def load_official_harmonizer_checkpoint(
    pipeline: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    audit_output: str | Path | None = None,
) -> LoadAudit:
    """Load official live DiT and VAE before A/B modules are attached."""

    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict_unet" not in payload:
        raise KeyError(f"Official checkpoint has no state_dict_unet: {path}")
    remapped = remap_official_unet_state_dict(payload["state_dict_unet"])
    live, ema = _split_live_and_ema(remapped)
    _require_ema_mirrors_live(live, ema)
    initial_audit = audit_official_dit_parameters(pipeline.dit, remapped)
    if initial_audit.missing_core or initial_audit.shape_mismatch:
        raise RuntimeError(
            "Official DiffusionHarmonizer live-DiT coverage is not 100%: "
            f"missing={len(initial_audit.missing_core)}, "
            f"shape_mismatch={len(initial_audit.shape_mismatch)}, "
            f"tensor_coverage={initial_audit.tensor_coverage:.6f}, "
            f"numel_coverage={initial_audit.numel_coverage:.6f}"
        )

    retained_runtime_state = load_official_live_dit_state(pipeline.dit, remapped)

    vae_state = payload.get("state_dict_vae")
    if vae_state is None:
        raise RuntimeError("Official DiffusionHarmonizer checkpoint has no state_dict_vae")
    vae_incompatible = inner_tokenizer_module(pipeline.tokenizer).load_state_dict(
        vae_state, strict=True
    )
    if vae_incompatible.missing_keys or vae_incompatible.unexpected_keys:
        raise RuntimeError("Strict VAE load unexpectedly returned incompatible keys")

    audit = LoadAudit(
        official_parameter_tensors=initial_audit.official_parameter_tensors,
        official_parameter_numel=initial_audit.official_parameter_numel,
        matched_parameter_tensors=initial_audit.matched_parameter_tensors,
        matched_parameter_numel=initial_audit.matched_parameter_numel,
        missing_core=initial_audit.missing_core,
        shape_mismatch=initial_audit.shape_mismatch,
        ignored_ema_tensors=initial_audit.ignored_ema_tensors,
        retained_runtime_state=retained_runtime_state,
        load_missing=(),
        load_unexpected=(),
    )
    if audit_output is not None:
        output = Path(audit_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n")
    return audit
