"""Synchronized, camera-isolated history for Scheme A-TMV inference.

The official DiffusionHarmonizer inference path constructs its temporal input
in this exact order::

    [current, output[-1], output[-2], output[-3], output[-4]]

and uses only ``[current]`` until *all* requested history offsets are
available.  This module applies that policy independently to every semantic
camera while advancing all camera queues in one atomic operation.

This manager represents one synchronized scene stream.  Tensors passed to it
have shape ``[num_views, ...]`` and :meth:`build_temporal_input` returns
``[num_frames, num_views, ...]``.  A caller that handles a batch of independent
scenes should use one manager per batch element and add/stack the batch axis
outside this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from numbers import Integral
from typing import Hashable, Sequence

import torch
from torch import Tensor


OFFICIAL_HISTORY_OFFSETS: tuple[int, ...] = (-1, -2, -3, -4)
"""Default historical-output offsets used by the official inference script."""

OFFICIAL_TEMPORAL_SLOT_OFFSETS: tuple[int, ...] = (
    0,
    *OFFICIAL_HISTORY_OFFSETS,
)
"""Full official slot order: current first, then newest-to-oldest history."""


@dataclass(frozen=True)
class HistoryAdvanceToken:
    """Opaque proof that an output belongs to a prepared scene/time step.

    Tokens are intentionally issued by :meth:`build_temporal_input` and
    consumed exactly once by :meth:`MultiViewHistoryManager.advance`.  The
    owner and generation fields prevent a late output from an old scene (or a
    different manager) from entering the active scene's queues.
    """

    _owner: object = field(repr=False, compare=True)
    generation: int
    synchronized_step: int
    build_serial: int
    camera_ids: tuple[int, ...]


@dataclass(frozen=True)
class TemporalHistoryBatch:
    """A prepared per-scene temporal input.

    Attributes:
        source: Tensor with axes ``[num_frames, num_views, ...]``.
        camera_ids: Semantic camera IDs in the source view-axis order.
        slot_offsets: Relative processed-time offsets for the frame axis.
            During cold start this is ``(0,)``.  Once ready it is
            ``(0, -1, -2, -3, -4)`` by default.
        camera_slot_steps: Absolute synchronized step index for every
            ``camera_ids``/``slot_offsets`` pair.  This metadata is validated
            on construction so a temporal slot can never silently represent
            different offsets for different cameras.
        scene_id: Active scene identifier.
        synchronized_step: Absolute processed-time index of ``source[0]``.
        advance_token: Single-use token required to commit the corresponding
            predicted outputs.
    """

    source: Tensor
    camera_ids: tuple[int, ...]
    slot_offsets: tuple[int, ...]
    camera_slot_steps: tuple[tuple[int, ...], ...]
    scene_id: Hashable | None
    synchronized_step: int
    advance_token: HistoryAdvanceToken = field(repr=False)

    def __post_init__(self) -> None:
        if self.source.ndim < 2:
            raise ValueError(
                "source must have at least [num_frames, num_views] axes, "
                f"got shape {tuple(self.source.shape)}"
            )
        if self.source.shape[0] != len(self.slot_offsets):
            raise ValueError("source frame axis does not match slot_offsets")
        if self.source.shape[1] != len(self.camera_ids):
            raise ValueError("source view axis does not match camera_ids")
        if len(self.camera_slot_steps) != len(self.camera_ids):
            raise ValueError("camera_slot_steps does not match camera_ids")
        self.validate_temporal_alignment()

    @property
    def num_frames(self) -> int:
        return len(self.slot_offsets)

    @property
    def num_views(self) -> int:
        return len(self.camera_ids)

    def validate_temporal_alignment(self) -> None:
        """Raise if any camera uses a different time for the same slot."""

        expected_steps = tuple(
            self.synchronized_step + offset for offset in self.slot_offsets
        )
        for camera_id, actual_steps in zip(
            self.camera_ids, self.camera_slot_steps, strict=True
        ):
            if actual_steps != expected_steps:
                raise RuntimeError(
                    "temporal slots are not synchronized for camera "
                    f"{camera_id}: expected steps {expected_steps}, "
                    f"got {actual_steps}"
                )


@dataclass(frozen=True)
class _HistoryEntry:
    output: Tensor
    synchronized_step: int


class MultiViewHistoryManager:
    """Maintain one output-history queue per semantic camera.

    ``build_temporal_input`` may automatically enter a new scene when its
    ``scene_id`` changes.  Entering a scene clears every camera queue and
    invalidates every outstanding advance token.  ``advance`` is atomic: it
    accepts one output for every configured camera, or modifies no queue.

    Args:
        camera_ids: Complete semantic camera set for this synchronized stream.
        history_offsets: Negative indices into prior predicted outputs.  Their
            supplied order is preserved exactly.  The default matches the
            official upstream inference behavior.
    """

    def __init__(
        self,
        camera_ids: Sequence[int] | Tensor,
        history_offsets: Sequence[int] = OFFICIAL_HISTORY_OFFSETS,
    ) -> None:
        self._camera_ids = self._normalize_configured_camera_ids(camera_ids)
        self._history_offsets = self._normalize_history_offsets(history_offsets)
        self._required_history = -min(self._history_offsets)
        self._queues: dict[int, deque[_HistoryEntry]] = {
            camera_id: deque(maxlen=self._required_history)
            for camera_id in self._camera_ids
        }

        self._owner = object()
        self._generation = 0
        self._build_serial = 0
        self._next_step = 0
        self._scene_id: Hashable | None = None
        self._scene_initialized = False
        self._pending_token: HistoryAdvanceToken | None = None

    @property
    def camera_ids(self) -> tuple[int, ...]:
        """Configured semantic camera IDs in canonical manager order."""

        return self._camera_ids

    @property
    def history_offsets(self) -> tuple[int, ...]:
        return self._history_offsets

    @property
    def full_slot_offsets(self) -> tuple[int, ...]:
        return (0, *self._history_offsets)

    @property
    def active_scene_id(self) -> Hashable | None:
        return self._scene_id

    @property
    def next_synchronized_step(self) -> int:
        return self._next_step

    @property
    def history_depth(self) -> int:
        """Number of retained synchronized outputs per camera."""

        self.validate_synchronized_history()
        return len(self._queues[self._camera_ids[0]])

    @property
    def ready(self) -> bool:
        """Whether every configured historical offset is currently valid."""

        return self.history_depth >= self._required_history

    @property
    def history_step_indices(self) -> dict[int, tuple[int, ...]]:
        """Oldest-to-newest retained synchronized step indices by camera."""

        return {
            camera_id: tuple(entry.synchronized_step for entry in queue)
            for camera_id, queue in self._queues.items()
        }

    def reset(self, scene_id: Hashable | None = None) -> None:
        """Clear all camera queues together and begin ``scene_id`` at step 0."""

        self._validate_scene_id(scene_id)
        for queue in self._queues.values():
            queue.clear()
        self._scene_id = scene_id
        self._scene_initialized = True
        self._next_step = 0
        self._generation += 1
        self._pending_token = None

    def build_temporal_input(
        self,
        current: Tensor,
        camera_ids: Sequence[int] | Tensor | None = None,
        *,
        scene_id: Hashable | None = None,
    ) -> TemporalHistoryBatch:
        """Prepare current inputs plus this scene's synchronized history.

        Args:
            current: Current source images/features with shape
                ``[num_views, ...]``.  Axis 0 follows ``camera_ids``.
            camera_ids: Semantic IDs for axis 0.  If omitted, manager order is
                used.  The complete configured set must be present exactly
                once, although its tensor order may change between steps.
            scene_id: Scene identity.  A change automatically and atomically
                resets all queues before constructing this input.

        Returns:
            A :class:`TemporalHistoryBatch`.  Its source frame order is
            current first followed by historical outputs in
            ``history_offsets`` order.  Until the deepest offset is available,
            source contains current only, exactly matching upstream cold start.
        """

        current = self._validate_view_tensor(current, "current")
        ordered_ids = self._normalize_runtime_camera_ids(camera_ids)
        self._validate_view_axis(current, ordered_ids, "current")
        self._activate_scene(scene_id)
        self.validate_synchronized_history()

        current_step = self._next_step
        if self.ready:
            slot_offsets = self.full_slot_offsets
            temporal_slots = [current]
            per_camera_steps: dict[int, list[int]] = {
                camera_id: [current_step] for camera_id in ordered_ids
            }

            for offset in self._history_offsets:
                expected_step = current_step + offset
                view_outputs: list[Tensor] = []
                for camera_id in ordered_ids:
                    entry = self._queues[camera_id][offset]
                    if entry.synchronized_step != expected_step:
                        raise RuntimeError(
                            "history offset is desynchronized for camera "
                            f"{camera_id}: offset {offset} at current step "
                            f"{current_step} should select step {expected_step}, "
                            f"got {entry.synchronized_step}"
                        )
                    view_outputs.append(entry.output)
                    per_camera_steps[camera_id].append(entry.synchronized_step)
                try:
                    temporal_slots.append(torch.stack(view_outputs, dim=0))
                except RuntimeError as error:
                    raise ValueError(
                        f"stored camera outputs for offset {offset} cannot be stacked"
                    ) from error

            try:
                source = torch.stack(temporal_slots, dim=0)
            except RuntimeError as error:
                raise ValueError(
                    "current input is incompatible with stored historical outputs "
                    "(shape, dtype, or device changed within a scene)"
                ) from error
        else:
            # Upstream does not construct a partially filled history window.
            slot_offsets = (0,)
            source = current.unsqueeze(0)
            per_camera_steps = {
                camera_id: [current_step] for camera_id in ordered_ids
            }

        self._build_serial += 1
        token = HistoryAdvanceToken(
            _owner=self._owner,
            generation=self._generation,
            synchronized_step=current_step,
            build_serial=self._build_serial,
            camera_ids=ordered_ids,
        )
        self._pending_token = token

        return TemporalHistoryBatch(
            source=source,
            camera_ids=ordered_ids,
            slot_offsets=slot_offsets,
            camera_slot_steps=tuple(
                tuple(per_camera_steps[camera_id]) for camera_id in ordered_ids
            ),
            scene_id=self._scene_id,
            synchronized_step=current_step,
            advance_token=token,
        )

    def advance(
        self,
        outputs: Tensor,
        camera_ids: Sequence[int] | Tensor | None = None,
        *,
        token: HistoryAdvanceToken,
    ) -> None:
        """Atomically append one synchronized predicted output per camera.

        ``token`` must come from the most recent
        :meth:`build_temporal_input` call.  It becomes invalid after one use,
        any reset/scene transition, or a newer build.  Outputs are detached
        and cloned so inference graphs and later caller-side mutations cannot
        alter retained history.
        """

        outputs = self._validate_view_tensor(outputs, "outputs")
        ordered_ids = self._normalize_runtime_camera_ids(camera_ids)
        self._validate_view_axis(outputs, ordered_ids, "outputs")
        self._validate_advance_token(token, ordered_ids)
        self.validate_synchronized_history()

        # Materialize every entry before touching a queue, preserving atomicity
        # if cloning fails for any camera.
        prepared_entries = {
            camera_id: _HistoryEntry(
                output=outputs[index].detach().clone(),
                synchronized_step=self._next_step,
            )
            for index, camera_id in enumerate(ordered_ids)
        }
        for camera_id in self._camera_ids:
            self._queues[camera_id].append(prepared_entries[camera_id])

        self._next_step += 1
        self._pending_token = None
        self.validate_synchronized_history()

    def validate_synchronized_history(self) -> None:
        """Validate equal queue depth and identical time positions for views."""

        step_sequences = self.history_step_indices
        reference_id = self._camera_ids[0]
        reference = step_sequences[reference_id]
        for camera_id in self._camera_ids[1:]:
            actual = step_sequences[camera_id]
            if actual != reference:
                raise RuntimeError(
                    "camera histories are not synchronized: camera "
                    f"{reference_id} has steps {reference}, camera {camera_id} "
                    f"has steps {actual}"
                )

        expected_start = self._next_step - len(reference)
        expected = tuple(range(expected_start, self._next_step))
        if reference != expected:
            raise RuntimeError(
                "history steps are not contiguous with the next synchronized "
                f"step {self._next_step}: expected {expected}, got {reference}"
            )

    def _activate_scene(self, scene_id: Hashable | None) -> None:
        self._validate_scene_id(scene_id)
        if not self._scene_initialized or scene_id != self._scene_id:
            self.reset(scene_id)

    def _validate_advance_token(
        self,
        token: HistoryAdvanceToken,
        ordered_ids: tuple[int, ...],
    ) -> None:
        if not isinstance(token, HistoryAdvanceToken):
            raise TypeError("token must be a HistoryAdvanceToken")
        if token._owner is not self._owner:
            raise RuntimeError("advance token belongs to a different history manager")
        if token.generation != self._generation:
            raise RuntimeError(
                "advance token belongs to an old scene generation; stale outputs "
                "must not enter the active scene"
            )
        if token.synchronized_step != self._next_step:
            raise RuntimeError(
                "advance token does not match the next synchronized step: "
                f"expected {self._next_step}, got {token.synchronized_step}"
            )
        if token.camera_ids != ordered_ids:
            raise ValueError(
                "outputs camera order must match the temporal input that issued "
                f"the token: expected {token.camera_ids}, got {ordered_ids}"
            )
        if self._pending_token != token:
            raise RuntimeError(
                "advance token is consumed or superseded by a newer temporal input"
            )

    def _normalize_runtime_camera_ids(
        self, camera_ids: Sequence[int] | Tensor | None
    ) -> tuple[int, ...]:
        if camera_ids is None:
            ordered_ids = self._camera_ids
        else:
            ordered_ids = self._normalize_camera_ids(camera_ids, "camera_ids")

        if len(ordered_ids) != len(self._camera_ids):
            raise ValueError(
                "camera_ids must contain every configured camera exactly once: "
                f"expected {self._camera_ids}, got {ordered_ids}"
            )
        if set(ordered_ids) != set(self._camera_ids):
            raise ValueError(
                "camera_ids semantic set differs from the configured cameras: "
                f"expected {self._camera_ids}, got {ordered_ids}"
            )
        return ordered_ids

    @classmethod
    def _normalize_configured_camera_ids(
        cls, camera_ids: Sequence[int] | Tensor
    ) -> tuple[int, ...]:
        normalized = cls._normalize_camera_ids(camera_ids, "camera_ids")
        if not normalized:
            raise ValueError("camera_ids must contain at least one camera")
        return normalized

    @staticmethod
    def _normalize_camera_ids(
        camera_ids: Sequence[int] | Tensor, name: str
    ) -> tuple[int, ...]:
        if isinstance(camera_ids, Tensor):
            if camera_ids.ndim != 1:
                raise ValueError(
                    f"{name} must be one-dimensional, got {tuple(camera_ids.shape)}"
                )
            if camera_ids.dtype != torch.long:
                raise TypeError(f"{name} tensor must have dtype torch.int64")
            values = camera_ids.detach().cpu().tolist()
        else:
            if isinstance(camera_ids, (str, bytes)):
                raise TypeError(f"{name} must be a sequence of integer IDs")
            values = list(camera_ids)

        if any(not isinstance(value, Integral) or isinstance(value, bool) for value in values):
            raise TypeError(f"{name} must contain only integer IDs")
        normalized = tuple(int(value) for value in values)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} must contain unique camera IDs")
        return normalized

    @staticmethod
    def _normalize_history_offsets(offsets: Sequence[int]) -> tuple[int, ...]:
        values = tuple(offsets)
        if not values:
            raise ValueError("history_offsets must contain at least one offset")
        if any(not isinstance(value, Integral) or isinstance(value, bool) for value in values):
            raise TypeError("history_offsets must contain only integers")
        normalized = tuple(int(value) for value in values)
        if any(value >= 0 for value in normalized):
            raise ValueError("history_offsets must be negative")
        if len(set(normalized)) != len(normalized):
            raise ValueError("history_offsets must not contain duplicates")
        return normalized

    @staticmethod
    def _validate_view_tensor(value: Tensor, name: str) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim < 1:
            raise ValueError(f"{name} must have a leading view axis")
        return value

    @staticmethod
    def _validate_view_axis(
        value: Tensor, camera_ids: tuple[int, ...], name: str
    ) -> None:
        if value.shape[0] != len(camera_ids):
            raise ValueError(
                f"{name} view axis has length {value.shape[0]}, but camera_ids "
                f"contains {len(camera_ids)} IDs"
            )

    @staticmethod
    def _validate_scene_id(scene_id: Hashable | None) -> None:
        try:
            hash(scene_id)
        except TypeError as error:
            raise TypeError("scene_id must be hashable") from error


__all__ = [
    "HistoryAdvanceToken",
    "MultiViewHistoryManager",
    "OFFICIAL_HISTORY_OFFSETS",
    "OFFICIAL_TEMPORAL_SLOT_OFFSETS",
    "TemporalHistoryBatch",
]
