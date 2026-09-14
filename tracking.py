from __future__ import annotations

"""Convert OmTrackVLA tracking rollouts to LeRobot v2.1.

The legacy seed-101 tar input and the processed four-view v1 input are both
supported.  Each source JSONL remains one complete episode.  Video frame ``t``
is the pre-action observation in row ``t``. ``observation.state[t] == action[t]``
is the first-frame-local cumulative nominal body pose, and row ``t``'s first
command advances pose ``t + 1``.
"""

import argparse
import hashlib
import io
import json
import logging
import math
import os
import shutil
import tarfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from statistics import median
from time import monotonic
from typing import Any, BinaryIO, Iterable, Sequence

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

TASK_DESCRIPTION_KEY = "annotation.human.action.task_description"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
POSE_AXES = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]

SEED = "seed_101"
FPS = 10
CONTROLLER_DT = 1.0 / 40.0
VELOCITY_SCALE = np.array([15.0, 10.0, 6.28], dtype=np.float64)
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
CHUNK_SIZE = 1000
VIDEO_KEY = "video.front"
STAGE4_SCHEMA_VERSION = "omtrackvla.processed_tracking.v1"
STAGE4_SOURCE_SCHEMA_VERSION = "omtrackvla.raw_episode.v1"
STAGE4_VIDEO_VIEW_MAP = {
    "video.front": "front",
    "video.left": "left",
    "video.right": "right",
    "video.rear": "back",
}
STAGE4_VIDEO_KEYS = tuple(STAGE4_VIDEO_VIEW_MAP)
VAL_UNSEEN_SALT = b"tracking-val-unseen-v1\0"
VAL_UNSEEN_INSTRUCTION_COUNT = 30
DEFAULT_JSONL_ARCHIVE = Path("/data4/glx/tracking/raw/archives/jsonl/seed_101.tar")
DEFAULT_FRAMES_DIR = Path("/data4/glx/tracking/raw/archives/frames/seed_101")
DEFAULT_OUTPUT_DIR = Path("/data4/glx/tracking/processed")
DEFAULT_WORK_DIR = Path("/data4/glx/tracking/work")
DEFAULT_FFMPEG = Path("/usr/local/bin/ffmpeg")

EXPECTED_FULL_COUNTS = {
    "train": {"episodes": 4924, "tasks": 593, "frames": 413087},
    "val_seen": {"episodes": 593, "tasks": 593, "frames": 43681},
    "val_unseen": {"episodes": 328, "tasks": 30, "frames": 25510},
}

PARQUET_COLUMNS = (
    TASK_DESCRIPTION_KEY,
    STATE_KEY,
    ACTION_KEY,
    "frame_index",
    "timestamp",
    "index",
    "episode_index",
    "task_index",
)


@dataclass(frozen=True)
class EpisodePlan:
    seed: str
    source_id: str
    source_episode_id: str
    numeric_episode_id: int
    jsonl_member: str
    instruction: str
    length: int
    split: str = ""
    episode_index: int = -1
    task_index: int = -1
    global_frame_start: int = -1

    @property
    def canonical_key(self) -> tuple[str, int, str]:
        return (self.source_id, self.numeric_episode_id, self.source_episode_id)

    @property
    def source_key(self) -> str:
        return f"{self.seed}/{self.source_id}/{self.source_episode_id}"


@dataclass(frozen=True)
class EpisodeResult:
    plan: EpisodePlan
    stats: dict[str, Any]
    extras: dict[str, Any]


@dataclass(frozen=True)
class Stage4EpisodeSource:
    """Lightweight metadata; each worker owns row and image validation."""

    plan: EpisodePlan
    raw_root: Path
    jsonl_path: Path
    camera_path: Path
    camera_metadata: dict[str, Any]


def configure_temp_environment(work_dir: str | Path) -> Path:
    """Keep this conversion and child processes away from the system /tmp."""
    temp_dir = Path(work_dir) / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    value = str(temp_dir.resolve())
    os.environ["TMPDIR"] = value
    os.environ["TEMP"] = value
    os.environ["TMP"] = value
    return temp_dir


def _parse_member_name(member_name: str) -> tuple[str, str, str, int]:
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 3:
        raise ValueError(f"invalid JSONL member path: {member_name!r}")
    seed, source_id, filename = path.parts
    if seed != SEED or not filename.endswith(".jsonl"):
        raise ValueError(f"unexpected JSONL member path: {member_name!r}")
    source_episode_id = filename[: -len(".jsonl")]
    try:
        numeric_episode_id = int(source_episode_id)
    except ValueError as exc:
        raise ValueError(f"non-numeric source episode id: {member_name!r}") from exc
    return seed, source_id, source_episode_id, numeric_episode_id


def _read_jsonl_rows(fileobj: BinaryIO, member_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(fileobj, 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {member_name}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"row is not an object at {member_name}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty episode JSONL: {member_name}")
    return rows


def _validate_row_schema(row: dict[str, Any], member_name: str, row_index: int) -> None:
    expected = {
        "images",
        "current",
        "instruction",
        "trajectory",
        "actions",
        "collision",
        "target_distance",
    }
    missing = expected.difference(row)
    if missing:
        raise ValueError(f"missing fields {sorted(missing)} at {member_name} row {row_index}")
    if not isinstance(row["instruction"], str) or not row["instruction"]:
        raise ValueError(f"invalid instruction at {member_name} row {row_index}")
    if not isinstance(row["current"], str) or not row["current"]:
        raise ValueError(f"invalid current path at {member_name} row {row_index}")
    actions = row["actions"]
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"missing actions at {member_name} row {row_index}")
    command = np.asarray(actions[0], dtype=np.float64)
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError(f"invalid actions[0] at {member_name} row {row_index}: {actions[0]!r}")


def scan_inventory(jsonl_archive: str | Path) -> list[EpisodePlan]:
    """Read source metadata without extracting it and build canonical episodes."""
    archive = Path(jsonl_archive)
    plans: list[EpisodePlan] = []
    seen_keys: set[tuple[str, int, str]] = set()
    with tarfile.open(archive, mode="r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            seed, source_id, source_episode_id, numeric_episode_id = _parse_member_name(member.name)
            extracted = tar.extractfile(member)
            if extracted is None:
                raise ValueError(f"failed to read JSONL member: {member.name}")
            rows = _read_jsonl_rows(extracted, member.name)
            instruction: str | None = None
            for row_index, row in enumerate(rows):
                _validate_row_schema(row, member.name, row_index)
                if instruction is None:
                    instruction = row["instruction"]
                elif row["instruction"] != instruction:
                    raise ValueError(
                        f"instruction changes inside {member.name}: row 0 != row {row_index}"
                    )
            assert instruction is not None
            plan = EpisodePlan(
                seed=seed,
                source_id=source_id,
                source_episode_id=source_episode_id,
                numeric_episode_id=numeric_episode_id,
                jsonl_member=member.name,
                instruction=instruction,
                length=len(rows),
            )
            if plan.canonical_key in seen_keys:
                raise ValueError(f"duplicate source episode key: {plan.canonical_key}")
            seen_keys.add(plan.canonical_key)
            plans.append(plan)
    if not plans:
        raise ValueError(f"no JSONL episodes found in {archive}")
    return sorted(plans, key=lambda item: item.canonical_key)


def instruction_hash(instruction: str) -> bytes:
    return hashlib.sha256(VAL_UNSEEN_SALT + instruction.encode("utf-8")).digest()


def split_inventory(
    plans: Sequence[EpisodePlan],
    val_unseen_instruction_count: int | None = None,
) -> list[EpisodePlan]:
    """Apply the frozen exact-instruction split without a manifest file."""
    grouped: dict[str, list[EpisodePlan]] = {}
    for plan in plans:
        grouped.setdefault(plan.instruction, []).append(plan)
    if val_unseen_instruction_count is None:
        val_unseen_instruction_count = min(
            VAL_UNSEEN_INSTRUCTION_COUNT,
            max(0, len(grouped) - 1),
        )
    if not 0 <= val_unseen_instruction_count < len(grouped):
        raise ValueError(
            "val_unseen_instruction_count must be non-negative and smaller than "
            f"the number of instructions ({len(grouped)})"
        )

    ranked_instructions = sorted(grouped, key=lambda text: (instruction_hash(text), text))
    unseen_instructions = set(ranked_instructions[:val_unseen_instruction_count])

    val_seen_keys: set[tuple[str, int, str]] = set()
    for instruction in ranked_instructions[val_unseen_instruction_count:]:
        candidates = grouped[instruction]
        group_median = float(median([item.length for item in candidates]))
        selected = min(
            candidates,
            key=lambda item: (abs(item.length - group_median), item.canonical_key),
        )
        val_seen_keys.add(selected.canonical_key)

    split_plans: list[EpisodePlan] = []
    for plan in plans:
        if plan.instruction in unseen_instructions:
            split = "val_unseen"
        elif plan.canonical_key in val_seen_keys:
            split = "val_seen"
        else:
            split = "train"
        split_plans.append(replace(plan, split=split))
    return split_plans


def assign_root_indices(plans: Sequence[EpisodePlan]) -> list[EpisodePlan]:
    """Preassign deterministic root-local episode/task/global frame indices."""
    assigned: list[EpisodePlan] = []
    for split in ("train", "val_seen", "val_unseen"):
        root_plans = sorted(
            (item for item in plans if item.split == split),
            key=lambda item: item.canonical_key,
        )
        task_indices: dict[str, int] = {}
        frame_start = 0
        for episode_index, plan in enumerate(root_plans):
            task_index = task_indices.setdefault(plan.instruction, len(task_indices))
            assigned.append(
                replace(
                    plan,
                    episode_index=episode_index,
                    task_index=task_index,
                    global_frame_start=frame_start,
                )
            )
            frame_start += plan.length
    return assigned


def summarize_plans(plans: Sequence[EpisodePlan]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "val_seen", "val_unseen"):
        selected = [item for item in plans if item.split == split]
        result[split] = {
            "episodes": len(selected),
            "tasks": len({item.instruction for item in selected}),
            "frames": sum(item.length for item in selected),
        }
    return result


def validate_full_split(plans: Sequence[EpisodePlan]) -> None:
    summary = summarize_plans(plans)
    if summary != EXPECTED_FULL_COUNTS:
        raise ValueError(f"full split totals differ from frozen contract: {summary}")
    instructions = {
        split: {item.instruction for item in plans if item.split == split}
        for split in ("train", "val_seen", "val_unseen")
    }
    if instructions["val_seen"] != instructions["train"]:
        raise ValueError("val_seen instruction set must exactly equal train")
    if instructions["val_unseen"] & (instructions["train"] | instructions["val_seen"]):
        raise ValueError("val_unseen instructions leak into train or val_seen")


def integrate_nominal_poses(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Build N pre-action poses from N rows; the final command is not integrated."""
    if not rows:
        raise ValueError("cannot integrate an empty episode")
    poses = np.empty((len(rows), 7), dtype=np.float32)
    poses[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    x = 0.0
    y = 0.0
    yaw = 0.0
    for row_index in range(len(rows) - 1):
        _validate_row_schema(rows[row_index], "<episode>", row_index)
        raw = np.asarray(rows[row_index]["actions"][0], dtype=np.float64)
        forward, left, yaw_rate = np.clip(raw, -1.0, 1.0) * VELOCITY_SCALE
        dx = (forward * math.cos(yaw) - left * math.sin(yaw)) * CONTROLLER_DT
        dy = (forward * math.sin(yaw) + left * math.cos(yaw)) * CONTROLLER_DT
        x += dx
        y += dy
        yaw += yaw_rate * CONTROLLER_DT
        poses[row_index + 1] = np.array(
            [x, y, 0.0, 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
            dtype=np.float32,
        )
    if not np.isfinite(poses).all():
        raise ValueError("nominal pose contains non-finite values")
    quaternion_norm = np.linalg.norm(poses[:, 3:7], axis=1)
    if not np.allclose(quaternion_norm, 1.0, atol=1e-5):
        raise ValueError("nominal pose contains a non-unit quaternion")
    return poses


def _root_for_split(output_dir: Path, split: str) -> Path:
    if split == "train":
        return output_dir / "train"
    if split in {"val_seen", "val_unseen"}:
        return output_dir / "test" / split
    raise ValueError(f"unknown split: {split!r}")


def _episode_paths(root: Path, episode_index: int) -> tuple[Path, Path]:
    chunk = episode_index // CHUNK_SIZE
    parquet_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    video_path = (
        root
        / "videos"
        / f"chunk-{chunk:03d}"
        / VIDEO_KEY
        / f"episode_{episode_index:06d}.mp4"
    )
    return parquet_path, video_path


def _episode_video_path(root: Path, episode_index: int, video_key: str) -> Path:
    chunk = episode_index // CHUNK_SIZE
    return (
        root
        / "videos"
        / f"chunk-{chunk:03d}"
        / video_key
        / f"episode_{episode_index:06d}.mp4"
    )


def prepare_layout(output_dir: Path, plans: Sequence[EpisodePlan]) -> None:
    for split in ("train", "val_seen", "val_unseen"):
        split_plans = [item for item in plans if item.split == split]
        if not split_plans:
            continue
        root = _root_for_split(output_dir, split)
        (root / "meta").mkdir(parents=True, exist_ok=True)
        (root / "depth").mkdir(parents=True, exist_ok=True)
        (root / "maps").mkdir(parents=True, exist_ok=True)
        total_chunks = math.ceil(len(split_plans) / CHUNK_SIZE)
        for chunk in range(total_chunks):
            (root / "data" / f"chunk-{chunk:03d}").mkdir(parents=True, exist_ok=True)
            video_chunk = root / "videos" / f"chunk-{chunk:03d}"
            for key in (VIDEO_KEY, "video.rear", "video.left", "video.right"):
                (video_chunk / key).mkdir(parents=True, exist_ok=True)


def _current_to_tar_member(current: str, plan: EpisodePlan) -> str:
    path = PurePosixPath(current)
    expected_prefix = ("frames", SEED, plan.source_id, plan.source_episode_id)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 5:
        raise ValueError(f"invalid current path in {plan.source_key}: {current!r}")
    if path.parts[:4] != expected_prefix:
        raise ValueError(
            f"current path does not match episode {plan.source_key}: {current!r}"
        )
    return PurePosixPath(*path.parts[2:]).as_posix()


def _fixed_list_array(values: np.ndarray, width: int, value_type: pa.DataType) -> pa.Array:
    flat = pa.array(values.reshape(-1), type=value_type)
    return pa.FixedSizeListArray.from_arrays(flat, width)


def write_episode_parquet(path: Path, plan: EpisodePlan, poses: np.ndarray) -> None:
    count = plan.length
    if poses.shape != (count, 7):
        raise ValueError(f"pose shape mismatch for {plan.source_key}: {poses.shape}")
    annotation_values = np.full((count, 1), plan.task_index, dtype=np.int32)
    frame_indices = np.arange(count, dtype=np.int64)
    arrays = [
        _fixed_list_array(annotation_values, 1, pa.int32()),
        _fixed_list_array(poses.astype(np.float32, copy=False), 7, pa.float32()),
        _fixed_list_array(poses.astype(np.float32, copy=True), 7, pa.float32()),
        pa.array(frame_indices, type=pa.int64()),
        pa.array(frame_indices.astype(np.float32) / np.float32(FPS), type=pa.float32()),
        pa.array(frame_indices + np.int64(plan.global_frame_start), type=pa.int64()),
        pa.array(np.full(count, plan.episode_index, dtype=np.int64), type=pa.int64()),
        pa.array(np.full(count, plan.task_index, dtype=np.int64), type=pa.int64()),
    ]
    table = pa.Table.from_arrays(arrays, names=PARQUET_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    metadata = pq.read_metadata(path)
    if metadata.num_rows != count:
        raise ValueError(f"Parquet row count mismatch at {path}: {metadata.num_rows} != {count}")
    written_schema = pq.read_schema(path)
    if tuple(written_schema.names) != PARQUET_COLUMNS:
        raise ValueError(f"Parquet columns mismatch at {path}: {written_schema.names}")


def _numeric_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values64 = np.asarray(values, dtype=np.float64)
    return {
        "min": values64.min(axis=0).tolist(),
        "max": values64.max(axis=0).tolist(),
        "mean": values64.mean(axis=0).tolist(),
        "std": values64.std(axis=0).tolist(),
        "count": [int(values64.shape[0])],
    }


class ImageStats:
    def __init__(self, *, width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT) -> None:
        self.width = int(width)
        self.height = int(height)
        self.minimum = np.full(3, np.inf, dtype=np.float64)
        self.maximum = np.full(3, -np.inf, dtype=np.float64)
        self.total = np.zeros(3, dtype=np.float64)
        self.square_total = np.zeros(3, dtype=np.float64)
        self.pixel_count = 0
        self.frame_count = 0

    def update(self, image: np.ndarray) -> None:
        if image.shape != (self.height, self.width, 3):
            raise ValueError(f"RGB image shape mismatch: {image.shape}")
        pixels = image.reshape(-1, 3).astype(np.float64) / 255.0
        self.minimum = np.minimum(self.minimum, pixels.min(axis=0))
        self.maximum = np.maximum(self.maximum, pixels.max(axis=0))
        self.total += pixels.sum(axis=0)
        self.square_total += np.square(pixels).sum(axis=0)
        self.pixel_count += pixels.shape[0]
        self.frame_count += 1

    def finalize(self) -> dict[str, list[float] | list[int]]:
        if self.frame_count == 0 or self.pixel_count == 0:
            raise ValueError("cannot finalize empty image stats")
        mean = self.total / self.pixel_count
        variance = np.maximum(self.square_total / self.pixel_count - np.square(mean), 0.0)
        return {
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [self.frame_count],
        }


def encode_video_from_tar(
    output_path: Path,
    image_members: Sequence[tarfile.TarInfo],
    frame_tar: tarfile.TarFile,
    ffmpeg_path: str | Path,
) -> dict[str, Any]:
    """Encode with PyAV's bundled libx264 directly from JPEG bytes."""
    del ffmpeg_path  # Kept in the public call contract for CLI compatibility.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_stats = ImageStats()
    try:
        with av.open(str(output_path), mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=FPS)
            stream.width = IMAGE_WIDTH
            stream.height = IMAGE_HEIGHT
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "veryfast", "crf": "23"}
            for member in image_members:
                extracted = frame_tar.extractfile(member)
                if extracted is None:
                    raise ValueError(f"failed to read frame member: {member.name}")
                data = extracted.read()
                with Image.open(io.BytesIO(data)) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                image_stats.update(rgb)
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"PyAV did not produce a video: {output_path}")
    return image_stats.finalize()


def validate_video(path: Path, expected_frames: int, decode: bool = True) -> dict[str, Any]:
    return validate_video_contract(
        path,
        expected_frames,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        decode=decode,
    )


def validate_video_contract(
    path: Path,
    expected_frames: int,
    *,
    width: int,
    height: int,
    decode: bool = True,
) -> dict[str, Any]:
    with av.open(str(path), mode="r") as container:
        video_streams = list(container.streams.video)
        if len(video_streams) != 1:
            raise ValueError(f"expected one video stream at {path}, got {len(video_streams)}")
        if list(container.streams.audio):
            raise ValueError(f"unexpected audio stream at {path}")
        stream = video_streams[0]
        codec = stream.codec_context.name
        pix_fmt = stream.codec_context.format.name if stream.codec_context.format else None
        rate = float(stream.average_rate) if stream.average_rate is not None else None
        if codec != "h264":
            raise ValueError(f"video codec mismatch at {path}: {codec}")
        if pix_fmt != "yuv420p":
            raise ValueError(f"video pixel format mismatch at {path}: {pix_fmt}")
        if stream.width != width or stream.height != height:
            raise ValueError(
                f"video resolution mismatch at {path}: {stream.width}x{stream.height}"
            )
        if rate is None or not math.isclose(rate, FPS, abs_tol=1e-6):
            raise ValueError(f"video FPS mismatch at {path}: {rate}")
        frame_count = sum(1 for _ in container.decode(stream)) if decode else int(stream.frames)
    if frame_count != expected_frames:
        raise ValueError(f"video frame count mismatch at {path}: {frame_count} != {expected_frames}")
    return {
        "codec": codec,
        "pix_fmt": pix_fmt,
        "fps": rate,
        "width": width,
        "height": height,
        "frames": frame_count,
    }


def encode_video_from_paths(
    output_path: Path,
    image_paths: Sequence[Path],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Encode a same-view JPEG sequence without resizing or changing cadence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_stats = ImageStats(width=width, height=height)
    try:
        with av.open(str(output_path), mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=FPS)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.codec_context.thread_count = 1
            stream.options = {"preset": "veryfast", "crf": "23", "threads": "1"}
            for image_path in image_paths:
                with Image.open(image_path) as image:
                    if image.size != (width, height):
                        raise ValueError(
                            f"image resolution mismatch at {image_path}: {image.size} != {(width, height)}"
                        )
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                image_stats.update(rgb)
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"PyAV did not produce a video: {output_path}")
    return image_stats.finalize()


def _process_episode(
    plan: EpisodePlan,
    rows: Sequence[dict[str, Any]],
    frame_tar: tarfile.TarFile,
    frame_members: dict[str, tarfile.TarInfo],
    physical_counts: dict[str, int],
    output_dir: Path,
    jsonl_archive: Path,
    frame_archive: Path,
    ffmpeg_path: Path,
) -> EpisodeResult:
    if len(rows) != plan.length:
        raise ValueError(f"row count changed for {plan.source_key}: {len(rows)} != {plan.length}")
    for row_index, row in enumerate(rows):
        _validate_row_schema(row, plan.jsonl_member, row_index)
        if row["instruction"] != plan.instruction:
            raise ValueError(f"instruction changed for {plan.source_key} at row {row_index}")

    member_names = [_current_to_tar_member(row["current"], plan) for row in rows]
    if len(set(member_names)) != len(member_names):
        raise ValueError(f"duplicate current frame in {plan.source_key}")
    image_members: list[tarfile.TarInfo] = []
    for name in member_names:
        member = frame_members.get(name)
        if member is None or not member.isfile():
            raise FileNotFoundError(f"missing current frame for {plan.source_key}: {name}")
        image_members.append(member)

    poses = integrate_nominal_poses(rows)
    root = _root_for_split(output_dir, plan.split)
    parquet_path, video_path = _episode_paths(root, plan.episode_index)
    parquet_partial = parquet_path.with_name(parquet_path.name + ".partial")
    video_partial = video_path.with_name(video_path.name + ".partial")
    for partial in (parquet_partial, video_partial):
        partial.unlink(missing_ok=True)
    try:
        write_episode_parquet(parquet_partial, plan, poses)
        video_stats = encode_video_from_tar(
            video_partial,
            image_members,
            frame_tar,
            ffmpeg_path,
        )
        validate_video(video_partial, plan.length, decode=True)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(parquet_partial, parquet_path)
        os.replace(video_partial, video_path)
    except BaseException:
        parquet_partial.unlink(missing_ok=True)
        video_partial.unlink(missing_ok=True)
        raise

    pose_stats = _numeric_stats(poses)
    stats = {
        STATE_KEY: pose_stats,
        VIDEO_KEY: video_stats,
        ACTION_KEY: pose_stats,
    }
    prefix = f"{plan.source_id}/{plan.source_episode_id}"
    extras = {
        "episode_index": plan.episode_index,
        "source_key": plan.source_key,
        "source_id": plan.source_id,
        "source_episode_id": plan.source_episode_id,
        "source_jsonl_archive": str(jsonl_archive),
        "source_jsonl_member": plan.jsonl_member,
        "source_frame_archive": str(frame_archive),
        "source_frame_prefix": prefix,
        "frame_count": plan.length,
        "physical_frame_count": physical_counts.get(prefix, 0),
        "fps": FPS,
        "capture_width": IMAGE_WIDTH,
        "capture_height": IMAGE_HEIGHT,
        "camera_keys": ["front"],
        "pose_semantics": "controller_nominal_first_frame_local_body_pose",
        "pose_translation_unit": "meter",
        "pose_rotation_unit": "radian",
        "pose_is_executed": False,
        "video.front.K": None,
        "video.front.body_from_camera": None,
        "K_front": None,
        "Extrinsic_front": None,
    }
    return EpisodeResult(plan=plan, stats=stats, extras=extras)


def _process_source(
    source_id: str,
    payloads: Sequence[tuple[EpisodePlan, list[dict[str, Any]]]],
    frames_dir: Path,
    output_dir: Path,
    jsonl_archive: Path,
    ffmpeg_path: Path,
) -> list[EpisodeResult]:
    frame_archive = frames_dir / f"{source_id}.tar"
    if not frame_archive.is_file():
        raise FileNotFoundError(f"missing frame archive: {frame_archive}")
    results: list[EpisodeResult] = []
    with tarfile.open(frame_archive, mode="r") as frame_tar:
        regular_members = [member for member in frame_tar.getmembers() if member.isfile()]
        frame_members = {member.name: member for member in regular_members}
        if len(frame_members) != len(regular_members):
            raise ValueError(f"duplicate frame member names in {frame_archive}")
        physical_counts: dict[str, int] = {}
        for member in regular_members:
            parts = PurePosixPath(member.name).parts
            if len(parts) >= 3:
                prefix = PurePosixPath(parts[0], parts[1]).as_posix()
                physical_counts[prefix] = physical_counts.get(prefix, 0) + 1
        for plan, rows in payloads:
            results.append(
                _process_episode(
                    plan,
                    rows,
                    frame_tar,
                    frame_members,
                    physical_counts,
                    output_dir,
                    jsonl_archive,
                    frame_archive,
                    ffmpeg_path,
                )
            )
    return results


def convert_episode_files(
    plans: Sequence[EpisodePlan],
    jsonl_archive: Path,
    frames_dir: Path,
    output_dir: Path,
    ffmpeg_path: Path,
    workers: int,
) -> list[EpisodeResult]:
    selected = {plan.jsonl_member: plan for plan in plans}
    if len(selected) != len(plans):
        raise ValueError("duplicate selected JSONL members")
    results: list[EpisodeResult] = []
    pending: set[Future[list[EpisodeResult]]] = set()
    completed_episodes = 0
    maximum_pending = max(1, workers * 2)

    def collect(done: Iterable[Future[list[EpisodeResult]]]) -> None:
        nonlocal completed_episodes
        for future in done:
            source_results = future.result()
            results.extend(source_results)
            completed_episodes += len(source_results)
            if completed_episodes % 100 < len(source_results) or completed_episodes == len(plans):
                logging.info("Converted %d/%d episodes", completed_episodes, len(plans))

    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="tracking") as executor:
        current_source: str | None = None
        payloads: list[tuple[EpisodePlan, list[dict[str, Any]]]] = []

        def submit_current() -> None:
            nonlocal payloads, pending
            if current_source is None or not payloads:
                payloads = []
                return
            pending.add(
                executor.submit(
                    _process_source,
                    current_source,
                    payloads,
                    frames_dir,
                    output_dir,
                    jsonl_archive,
                    ffmpeg_path,
                )
            )
            payloads = []
            if len(pending) >= maximum_pending:
                done, remaining = wait(pending, return_when=FIRST_COMPLETED)
                pending = set(remaining)
                collect(done)

        with tarfile.open(jsonl_archive, mode="r") as json_tar:
            for member in json_tar:
                if not member.isfile():
                    continue
                _, source_id, _, _ = _parse_member_name(member.name)
                if current_source is None:
                    current_source = source_id
                elif source_id != current_source:
                    submit_current()
                    current_source = source_id
                plan = selected.get(member.name)
                if plan is None:
                    continue
                extracted = json_tar.extractfile(member)
                if extracted is None:
                    raise ValueError(f"failed to read selected JSONL member: {member.name}")
                payloads.append((plan, _read_jsonl_rows(extracted, member.name)))
            submit_current()
        while pending:
            done, remaining = wait(pending, return_when=FIRST_COMPLETED)
            pending = set(remaining)
            collect(done)

    if len(results) != len(plans):
        converted = {result.plan.jsonl_member for result in results}
        missing = sorted(set(selected).difference(converted))
        raise ValueError(f"conversion result count mismatch; missing={missing[:10]}")
    return results


def _json_dump(value: Any, *, indent: int | None = None) -> str:
    separators = None if indent is not None else (",", ":")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        separators=separators,
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(_json_dump(row))
            file.write("\n")


def _features(
    video_specs: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Build a LeRobot feature contract from ``video_key -> (width, height)``."""
    if video_specs is None:
        video_specs = {VIDEO_KEY: (IMAGE_WIDTH, IMAGE_HEIGHT)}
    pose_feature = {
        "dtype": "float32",
        "shape": [7],
        "names": {"axes": POSE_AXES},
    }
    features: dict[str, Any] = {
        TASK_DESCRIPTION_KEY: {"dtype": "int32", "shape": [1], "names": None},
        STATE_KEY: pose_feature,
    }
    for video_key, (width, height) in video_specs.items():
        features[video_key] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": FPS,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    features.update({
        ACTION_KEY: pose_feature,
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    })
    return features


def _modality(video_keys: Sequence[str] | None = None) -> dict[str, Any]:
    if video_keys is None:
        video_keys = (VIDEO_KEY,)
    return {
        "state": {
            "drone": {"start": 0, "end": 7, "original_key": STATE_KEY},
        },
        "action": {
            "state": {"start": 0, "end": 7, "absolute": True, "original_key": ACTION_KEY},
        },
        "video": {
            key.removeprefix("video."): {"original_key": key}
            for key in video_keys
        },
        "annotation": {
            TASK_DESCRIPTION_KEY: {"original_key": TASK_DESCRIPTION_KEY},
        },
        "extra": [],
    }


def write_metadata(
    output_dir: Path,
    plans: Sequence[EpisodePlan],
    results: Sequence[EpisodeResult],
    *,
    video_specs: dict[str, tuple[int, int]] | None = None,
) -> None:
    if video_specs is None:
        video_specs = {VIDEO_KEY: (IMAGE_WIDTH, IMAGE_HEIGHT)}
    results_by_member = {result.plan.jsonl_member: result for result in results}
    for split in ("train", "val_seen", "val_unseen"):
        root_plans = sorted(
            (plan for plan in plans if plan.split == split),
            key=lambda item: item.episode_index,
        )
        if not root_plans:
            continue
        root = _root_for_split(output_dir, split)
        meta = root / "meta"
        tasks_by_index: dict[int, str] = {}
        for plan in root_plans:
            existing = tasks_by_index.setdefault(plan.task_index, plan.instruction)
            if existing != plan.instruction:
                raise ValueError(f"task index collision in {split}: {plan.task_index}")
        if sorted(tasks_by_index) != list(range(len(tasks_by_index))):
            raise ValueError(f"non-contiguous task indices in {split}")

        total_frames = sum(plan.length for plan in root_plans)
        info = {
            "codebase_version": "v2.1",
            "robot_type": "spot",
            "fps": FPS,
            "total_episodes": len(root_plans),
            "total_frames": total_frames,
            "total_tasks": len(tasks_by_index),
            "total_videos": len(root_plans) * len(video_specs),
            "total_chunks": math.ceil(len(root_plans) / CHUNK_SIZE),
            "chunks_size": CHUNK_SIZE,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": _features(video_specs),
            "splits": {"train": f"0:{len(root_plans)}"},
        }
        (meta / "info.json").write_text(_json_dump(info, indent=4) + "\n", encoding="utf-8")
        (meta / "modality.json").write_text(
            _json_dump(_modality(tuple(video_specs)), indent=2) + "\n", encoding="utf-8"
        )
        _write_jsonl(
            meta / "tasks.jsonl",
            ({"task_index": index, "task": tasks_by_index[index]} for index in sorted(tasks_by_index)),
        )
        _write_jsonl(
            meta / "episodes.jsonl",
            (
                {
                    "episode_index": plan.episode_index,
                    "tasks": [plan.instruction],
                    "length": plan.length,
                }
                for plan in root_plans
            ),
        )
        _write_jsonl(
            meta / "episodes_stats.jsonl",
            (
                {
                    "episode_index": plan.episode_index,
                    "stats": results_by_member[plan.jsonl_member].stats,
                }
                for plan in root_plans
            ),
        )
        _write_jsonl(
            meta / "episodes_extras.jsonl",
            (results_by_member[plan.jsonl_member].extras for plan in root_plans),
        )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def validate_output_root(root: Path, decode_videos: bool = False) -> dict[str, int]:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks = _load_jsonl(root / "meta" / "tasks.jsonl")
    episodes = _load_jsonl(root / "meta" / "episodes.jsonl")
    stats = _load_jsonl(root / "meta" / "episodes_stats.jsonl")
    extras = _load_jsonl(root / "meta" / "episodes_extras.jsonl")
    total_episodes = int(info["total_episodes"])
    if not (len(episodes) == len(stats) == len(extras) == total_episodes):
        raise ValueError(f"metadata line counts differ at {root}")
    if len(tasks) != int(info["total_tasks"]):
        raise ValueError(f"task count differs at {root}")
    video_specs: dict[str, tuple[int, int]] = {}
    for key, feature in info["features"].items():
        if isinstance(feature, dict) and feature.get("dtype") == "video":
            shape = feature.get("shape")
            if not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3:
                raise ValueError(f"invalid video feature shape at {root}: {key}={shape!r}")
            video_specs[key] = (int(shape[1]), int(shape[0]))
    if not video_specs:
        raise ValueError(f"no video features at {root}")
    if info["features"] != _features(video_specs):
        raise ValueError(f"feature contract differs at {root}")
    if int(info.get("total_videos", -1)) != total_episodes * len(video_specs):
        raise ValueError(f"total_videos differs at {root}")
    if json.loads((root / "meta" / "modality.json").read_text(encoding="utf-8")) != _modality(
        tuple(video_specs)
    ):
        raise ValueError(f"modality contract differs at {root}")

    expected_global_index = 0
    total_frames = 0
    for expected_episode_index, episode in enumerate(episodes):
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        if episode_index != expected_episode_index:
            raise ValueError(f"non-contiguous episode index at {root}: {episode_index}")
        parquet_path, video_path = _episode_paths(root, episode_index)
        table = pq.read_table(parquet_path)
        if tuple(table.column_names) != PARQUET_COLUMNS or table.num_rows != length:
            raise ValueError(f"Parquet contract mismatch: {parquet_path}")
        frame_index = table["frame_index"].to_numpy()
        global_index = table["index"].to_numpy()
        timestamps = table["timestamp"].to_numpy()
        if not np.array_equal(frame_index, np.arange(length, dtype=np.int64)):
            raise ValueError(f"frame_index mismatch: {parquet_path}")
        if not np.array_equal(
            global_index,
            np.arange(expected_global_index, expected_global_index + length, dtype=np.int64),
        ):
            raise ValueError(f"global index mismatch: {parquet_path}")
        if not np.allclose(
            timestamps,
            np.arange(length, dtype=np.float32) / np.float32(FPS),
            atol=1e-7,
        ):
            raise ValueError(f"timestamp mismatch: {parquet_path}")
        state = np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32)
        action = np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32)
        if not np.array_equal(state, action):
            raise ValueError(f"state/action mismatch: {parquet_path}")
        if not np.allclose(state[0], [0, 0, 0, 0, 0, 0, 1], atol=1e-6):
            raise ValueError(f"first pose is not identity: {parquet_path}")
        if not np.isfinite(state).all() or not np.allclose(
            np.linalg.norm(state[:, 3:7], axis=1), 1.0, atol=1e-5
        ):
            raise ValueError(f"invalid pose values: {parquet_path}")
        del video_path
        for video_key, (width, height) in video_specs.items():
            validate_video_contract(
                _episode_video_path(root, episode_index, video_key),
                length,
                width=width,
                height=height,
                decode=decode_videos,
            )
        extra = extras[episode_index]
        if extra["episode_index"] != episode_index:
            raise ValueError(f"extras index mismatch at {root}: {episode_index}")
        camera_keys = extra.get("camera_keys")
        if tuple(video_specs) == (VIDEO_KEY,) and camera_keys == ["front"]:
            for key in (
                "video.front.K",
                "video.front.body_from_camera",
                "K_front",
                "Extrinsic_front",
            ):
                if extra.get(key, "missing") is not None:
                    raise ValueError(f"calibration must be null in {root}: {key}")
        else:
            expected_camera_keys = [key.removeprefix("video.") for key in video_specs]
            if camera_keys != expected_camera_keys:
                raise ValueError(
                    f"camera key mismatch at {root}: {camera_keys!r} != {expected_camera_keys!r}"
                )
            for video_key in video_specs:
                short_key = video_key.removeprefix("video.")
                intrinsic = np.asarray(extra.get(f"{video_key}.K"), dtype=np.float64)
                extrinsic = np.asarray(
                    extra.get(f"{video_key}.body_from_camera"), dtype=np.float64
                )
                if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
                    raise ValueError(f"invalid intrinsic in {root}: {video_key}")
                if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
                    raise ValueError(f"invalid extrinsic in {root}: {video_key}")
                if extra.get(f"K_{short_key}") != extra.get(f"{video_key}.K"):
                    raise ValueError(f"intrinsic alias mismatch in {root}: {video_key}")
                if extra.get(f"Extrinsic_{short_key}") != extra.get(
                    f"{video_key}.body_from_camera"
                ):
                    raise ValueError(f"extrinsic alias mismatch in {root}: {video_key}")
        expected_global_index += length
        total_frames += length
    if total_frames != int(info["total_frames"]):
        raise ValueError(f"total frame count differs at {root}")
    return {"episodes": total_episodes, "tasks": len(tasks), "frames": total_frames}


def validate_output_dataset(output_dir: str | Path, decode_videos: bool = False) -> dict[str, Any]:
    output = Path(output_dir)
    summary: dict[str, Any] = {}
    for split in ("train", "val_seen", "val_unseen"):
        root = _root_for_split(output, split)
        if root.exists():
            summary[split] = validate_output_root(root, decode_videos=decode_videos)
    if not summary:
        raise ValueError(f"no LeRobot roots found under {output}")
    if set(summary) == set(EXPECTED_FULL_COUNTS) and all(
        summary[split]["episodes"] == EXPECTED_FULL_COUNTS[split]["episodes"]
        for split in EXPECTED_FULL_COUNTS
    ) and summary != EXPECTED_FULL_COUNTS:
        raise ValueError(f"output totals differ from frozen full contract: {summary}")
    return summary


def _resolve_stage4_relative(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid {label}: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    resolved = (root / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes processed root: {value!r}") from exc
    return resolved


def _stage4_camera_contract(
    camera: dict[str, Any],
    camera_path: Path,
) -> dict[str, tuple[int, int]]:
    if camera.get("schema_version") != STAGE4_SOURCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported camera schema at {camera_path}")
    timing = camera.get("timing")
    if not isinstance(timing, dict):
        raise ValueError(f"missing timing metadata at {camera_path}")
    expected_timing = {
        "controller_hz": 40.0,
        "action_repeat": 4.0,
        "frame_hz": float(FPS),
        "frame_dt_s": 1.0 / float(FPS),
        "controller_integration_dt_s": CONTROLLER_DT,
        "video_fps": float(FPS),
    }
    for key, expected in expected_timing.items():
        try:
            actual = float(timing[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid timing field {key!r} at {camera_path}") from exc
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"timing field {key!r} differs at {camera_path}: {actual} != {expected}"
            )
    if timing.get("frame_semantics") != "pre_action_observation":
        raise ValueError(f"unsupported frame semantics at {camera_path}")

    source_views = camera.get("views")
    if not isinstance(source_views, dict):
        raise ValueError(f"missing camera views at {camera_path}")
    video_specs: dict[str, tuple[int, int]] = {}
    for video_key, source_view in STAGE4_VIDEO_VIEW_MAP.items():
        view = source_views.get(source_view)
        if not isinstance(view, dict):
            raise ValueError(f"missing {source_view!r} camera at {camera_path}")
        try:
            width = int(view["width_px"])
            height = int(view["height_px"])
            hfov = float(view["hfov_deg"])
            intrinsic = np.asarray(view["K"], dtype=np.float64)
            extrinsic = np.asarray(view["body_from_camera"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid {source_view!r} camera at {camera_path}") from exc
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError(
                f"camera resolution must be positive and even at {camera_path}: {width}x{height}"
            )
        if not 0.0 < hfov < 180.0:
            raise ValueError(f"invalid HFOV for {source_view!r} at {camera_path}: {hfov}")
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError(f"invalid K for {source_view!r} at {camera_path}")
        if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
            raise ValueError(f"invalid body_from_camera for {source_view!r} at {camera_path}")
        if not np.allclose(extrinsic[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError(f"invalid homogeneous transform for {source_view!r} at {camera_path}")
        video_specs[video_key] = (width, height)
    return video_specs


def scan_stage4_inventory(
    processed_root: str | Path,
) -> tuple[list[Stage4EpisodeSource], dict[str, tuple[int, int]]]:
    """Index metadata and first-row instructions without scanning images or full JSONLs."""
    root = Path(processed_root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != STAGE4_SCHEMA_VERSION:
        raise ValueError(f"unsupported processed schema at {manifest_path}")
    raw_root_value = manifest.get("input_root")
    if not isinstance(raw_root_value, str) or not raw_root_value:
        raise ValueError(f"missing input_root provenance in {manifest_path}")
    raw_root_path = Path(raw_root_value)
    if not raw_root_path.is_absolute():
        raise ValueError(f"input_root must be absolute in {manifest_path}")
    raw_root = raw_root_path.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    successful = manifest.get("successful_episodes")
    if not isinstance(successful, list) or not successful:
        raise ValueError(f"no successful episodes in {manifest_path}")

    preliminary: list[Stage4EpisodeSource] = []
    common_video_specs: dict[str, tuple[int, int]] | None = None
    seen_source_episodes: set[str] = set()
    entries = sorted(successful, key=lambda item: str(item.get("source_episode", "")))
    for ordinal, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"invalid successful episode entry in {manifest_path}")
        source_episode = item.get("source_episode")
        if not isinstance(source_episode, str):
            raise ValueError(f"invalid source_episode in {manifest_path}: {source_episode!r}")
        source_parts = PurePosixPath(source_episode).parts
        if len(source_parts) != 3 or ".." in source_parts:
            raise ValueError(f"invalid source_episode: {source_episode!r}")
        if source_episode in seen_source_episodes:
            raise ValueError(f"duplicate source_episode: {source_episode}")
        seen_source_episodes.add(source_episode)
        seed, source_id, source_episode_id = source_parts

        source_manifest_path = _resolve_stage4_relative(
            root, item.get("manifest"), "source manifest path"
        )
        if not source_manifest_path.is_file():
            raise FileNotFoundError(source_manifest_path)
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest.get("schema_version") != STAGE4_SCHEMA_VERSION:
            raise ValueError(f"unsupported source manifest at {source_manifest_path}")
        if source_manifest.get("source_episode") != source_episode:
            raise ValueError(f"source episode mismatch at {source_manifest_path}")
        outputs = source_manifest.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError(f"missing outputs at {source_manifest_path}")
        jsonl_output = outputs.get("jsonl")
        if not isinstance(jsonl_output, dict):
            raise ValueError(f"missing JSONL output at {source_manifest_path}")
        jsonl_path = _resolve_stage4_relative(root, jsonl_output.get("path"), "JSONL path")
        if not jsonl_path.is_file():
            raise FileNotFoundError(jsonl_path)
        length = int(source_manifest.get("sample_count", -1))
        if length <= 0:
            raise ValueError(f"invalid sample count at {source_manifest_path}: {length}")
        with jsonl_path.open("r", encoding="utf-8") as file:
            first_row = next((json.loads(line) for line in file if line.strip()), None)
        if not isinstance(first_row, dict):
            raise ValueError(f"missing first row at {jsonl_path}")
        instruction = first_row.get("instruction")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError(f"invalid instruction at {jsonl_path}")

        camera_path = _resolve_stage4_relative(
            root, outputs.get("camera_metadata"), "camera metadata path"
        )
        if not camera_path.is_file():
            raise FileNotFoundError(camera_path)
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        video_specs = _stage4_camera_contract(camera, camera_path)
        if common_video_specs is None:
            common_video_specs = video_specs
        elif video_specs != common_video_specs:
            raise ValueError(
                f"camera resolutions differ across processed episodes: {camera_path}"
            )

        suffix = source_episode_id.rsplit("_", 1)[-1]
        numeric_episode_id = int(suffix) if suffix.isdigit() else ordinal
        plan = EpisodePlan(
            seed=seed,
            source_id=source_id,
            source_episode_id=source_episode_id,
            numeric_episode_id=numeric_episode_id,
            jsonl_member=jsonl_path.relative_to(root).as_posix(),
            instruction=instruction,
            length=length,
            split="train",
        )
        preliminary.append(
            Stage4EpisodeSource(
                plan=plan,
                raw_root=raw_root,
                jsonl_path=jsonl_path,
                camera_path=camera_path,
                camera_metadata=camera,
            )
        )

        if (ordinal + 1) % 100 == 0 or ordinal + 1 == len(entries):
            logging.info("Indexed metadata for %d/%d episodes", ordinal + 1, len(entries))

    assert common_video_specs is not None
    assigned = {
        plan.jsonl_member: plan
        for plan in assign_root_indices([source.plan for source in preliminary])
    }
    sources = [replace(source, plan=assigned[source.plan.jsonl_member]) for source in preliminary]
    sources.sort(key=lambda source: source.plan.episode_index)
    return sources, common_video_specs


def _load_stage4_episode(
    source: Stage4EpisodeSource,
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[Path]]]:
    """Validate one episode in its worker; image content is checked during encoding."""
    plan = source.plan
    seed, source_id, source_episode_id = plan.seed, plan.source_id, plan.source_episode_id
    jsonl_path = source.jsonl_path
    rows = _load_jsonl(jsonl_path)
    if len(rows) != plan.length:
        raise ValueError(f"sample count mismatch at {jsonl_path}: {len(rows)} != {plan.length}")
    resolved_camera_paths: dict[str, Path] = {}
    image_paths: dict[str, list[Path]] = {key: [] for key in STAGE4_VIDEO_KEYS}
    seen_view_paths: dict[str, set[Path]] = {key: set() for key in STAGE4_VIDEO_KEYS}
    for row_index, row in enumerate(rows):
        _validate_row_schema(row, str(jsonl_path), row_index)
        if row.get("episode_id") != source_episode_id:
            raise ValueError(f"episode id mismatch at {jsonl_path} row {row_index}")
        if row.get("frame_index") != row_index:
            raise ValueError(f"frame_index mismatch at {jsonl_path} row {row_index}")
        try:
            sim_time = float(row["sim_time_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid sim_time_s at {jsonl_path} row {row_index}") from exc
        if not math.isclose(sim_time, row_index / FPS, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"10 Hz sim_time_s mismatch at {jsonl_path} row {row_index}")
        if row["instruction"] != plan.instruction:
            raise ValueError(f"instruction changes at {jsonl_path} row {row_index}")
        camera_reference = row.get("camera_metadata")
        if not isinstance(camera_reference, str) or not camera_reference:
            raise ValueError(f"invalid row camera metadata path at {jsonl_path} row {row_index}")
        if camera_reference not in resolved_camera_paths:
            resolved_camera_paths[camera_reference] = _resolve_stage4_relative(
                root, camera_reference, "row camera metadata path"
            )
        if resolved_camera_paths[camera_reference] != source.camera_path:
            raise ValueError(f"camera metadata mismatch at {jsonl_path} row {row_index}")
        current_views = row.get("current_views")
        if not isinstance(current_views, dict):
            raise ValueError(f"missing current_views at {jsonl_path} row {row_index}")
        for video_key, source_view in STAGE4_VIDEO_VIEW_MAP.items():
            image_path = _resolve_stage4_relative(
                root,
                current_views.get(source_view),
                f"{source_view} image path",
            )
            expected_prefix = ("frames", seed, source_id, source_episode_id, source_view)
            relative_parts = image_path.relative_to(root).parts
            if len(relative_parts) != 6 or relative_parts[:5] != expected_prefix:
                raise ValueError(
                    f"{source_view} image does not match episode at {jsonl_path} row {row_index}"
                )
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if image_path in seen_view_paths[video_key]:
                raise ValueError(f"duplicate {source_view} image at {jsonl_path}")
            seen_view_paths[video_key].add(image_path)
            image_paths[video_key].append(image_path)
        front_relative = image_paths[VIDEO_KEY][-1].relative_to(root).as_posix()
        if row["current"] != front_relative:
            raise ValueError(f"current is not current_views.front at {jsonl_path} row {row_index}")

    return rows, image_paths


def _process_stage4_episode(
    source: Stage4EpisodeSource,
    processed_root: Path,
    output_dir: Path,
    video_specs: dict[str, tuple[int, int]],
) -> EpisodeResult:
    plan = source.plan
    rows, image_paths = _load_stage4_episode(source, processed_root)
    poses = integrate_nominal_poses(rows)
    root = _root_for_split(output_dir, plan.split)
    parquet_path, _ = _episode_paths(root, plan.episode_index)
    video_paths = {
        key: _episode_video_path(root, plan.episode_index, key) for key in video_specs
    }
    parquet_partial = parquet_path.with_name(parquet_path.name + ".partial")
    video_partials = {
        key: path.with_name(path.name + ".partial") for key, path in video_paths.items()
    }
    for partial in (parquet_partial, *video_partials.values()):
        partial.unlink(missing_ok=True)
    video_stats: dict[str, dict[str, Any]] = {}
    try:
        write_episode_parquet(parquet_partial, plan, poses)
        for video_key, (width, height) in video_specs.items():
            video_stats[video_key] = encode_video_from_paths(
                video_partials[video_key],
                image_paths[video_key],
                width=width,
                height=height,
            )
            validate_video_contract(
                video_partials[video_key],
                plan.length,
                width=width,
                height=height,
                decode=True,
            )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(parquet_partial, parquet_path)
        for video_key, final_path in video_paths.items():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(video_partials[video_key], final_path)
    except BaseException:
        for partial in (parquet_partial, *video_partials.values()):
            partial.unlink(missing_ok=True)
        raise

    pose_stats = _numeric_stats(poses)
    stats = {STATE_KEY: pose_stats, **video_stats, ACTION_KEY: pose_stats}
    timing = source.camera_metadata["timing"]
    front_width, front_height = video_specs[VIDEO_KEY]
    extras: dict[str, Any] = {
        "episode_index": plan.episode_index,
        "source_key": plan.source_key,
        "source_id": plan.source_id,
        "source_episode_id": plan.source_episode_id,
        "source_raw_root": str(source.raw_root),
        "source_raw_episode": plan.source_key,
        "source_raw_episode_path": str(source.raw_root / Path(*PurePosixPath(plan.source_key).parts)),
        "source_intermediate_schema_version": STAGE4_SCHEMA_VERSION,
        "source_intermediate_jsonl": source.jsonl_path.relative_to(processed_root).as_posix(),
        "source_intermediate_camera_metadata": source.camera_path.relative_to(processed_root).as_posix(),
        "frame_count": plan.length,
        "fps": FPS,
        "frame_dt_s": 1.0 / FPS,
        "controller_hz": float(timing["controller_hz"]),
        "action_repeat": int(timing["action_repeat"]),
        "controller_integration_dt_s": float(timing["controller_integration_dt_s"]),
        "capture_width": front_width,
        "capture_height": front_height,
        "camera_keys": [key.removeprefix("video.") for key in video_specs],
        "source_camera_keys": [STAGE4_VIDEO_VIEW_MAP[key] for key in video_specs],
        "source_to_output_view": {
            source_view: video_key.removeprefix("video.")
            for video_key, source_view in STAGE4_VIDEO_VIEW_MAP.items()
        },
        "pose_semantics": "controller_nominal_first_frame_local_body_pose",
        "pose_translation_unit": "meter",
        "pose_rotation_unit": "radian",
        "pose_is_executed": False,
    }
    camera_views = source.camera_metadata["views"]
    for video_key, source_view in STAGE4_VIDEO_VIEW_MAP.items():
        short_key = video_key.removeprefix("video.")
        intrinsic = camera_views[source_view]["K"]
        extrinsic = camera_views[source_view]["body_from_camera"]
        extras[f"{video_key}.K"] = intrinsic
        extras[f"{video_key}.body_from_camera"] = extrinsic
        extras[f"K_{short_key}"] = intrinsic
        extras[f"Extrinsic_{short_key}"] = extrinsic
    return EpisodeResult(plan=plan, stats=stats, extras=extras)


def convert_stage4_dataset(
    processed_root: str | Path,
    output_dir: str | Path,
    work_dir: str | Path = DEFAULT_WORK_DIR,
    *,
    workers: int = 4,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert processed four-view v1 data directly into one LeRobot train root."""
    processed_root = Path(processed_root).resolve()
    output_dir = Path(output_dir).resolve()
    work_dir = Path(work_dir).resolve()
    configure_temp_environment(work_dir)
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not processed_root.is_dir():
        raise FileNotFoundError(processed_root)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {output_dir}")

    logging.info("Reading processed four-view metadata: %s", processed_root)
    sources, video_specs = scan_stage4_inventory(processed_root)
    plans = [source.plan for source in sources]
    staging_dir = work_dir / f"{output_dir.name}.stage4-staging-{os.getpid()}"
    if staging_dir.exists():
        raise FileExistsError(f"staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    try:
        prepare_layout(staging_dir, plans)
        total_frames = sum(plan.length for plan in plans)
        worker_count = min(workers, len(sources))
        logging.info(
            "Converting %d episodes / %d frames with %d workers (including input validation)",
            len(sources), total_frames, worker_count,
        )
        started = monotonic()
        completed_frames = 0
        results: list[EpisodeResult] = []
        with ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="tracking-stage4"
        ) as executor:
            futures = [
                executor.submit(
                    _process_stage4_episode,
                    source,
                    processed_root,
                    staging_dir,
                    video_specs,
                )
                for source in sources
            ]
            try:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    completed_frames += result.plan.length
                    completed = len(results)
                    if completed == 1 or completed % 25 == 0 or completed == len(sources):
                        rate = completed_frames / max(monotonic() - started, 1e-6)
                        logging.info(
                            "Converted %d/%d episodes, %d/%d frames, %.1f frames/s, "
                            "conversion ETA %.0fs (final metadata checks follow)",
                            completed, len(sources), completed_frames, total_frames,
                            rate, (total_frames - completed_frames) / rate,
                        )
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        write_metadata(staging_dir, plans, results, video_specs=video_specs)
        logging.info("Validating final dataset metadata: %s", staging_dir)
        summary = validate_output_dataset(staging_dir, decode_videos=False)
        expected_summary = {
            "train": {
                "episodes": len(plans),
                "tasks": len({plan.instruction for plan in plans}),
                "frames": sum(plan.length for plan in plans),
            }
        }
        if summary != expected_summary:
            raise ValueError(f"processed four-view conversion summary mismatch: {summary}")
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(output_dir)
            shutil.rmtree(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_dir, output_dir)
    except BaseException:
        logging.error("Conversion failed; staging data retained at %s", staging_dir)
        raise
    return {
        "input_format": STAGE4_SCHEMA_VERSION,
        "processed_root": str(processed_root),
        "output_dir": str(output_dir),
        "video_keys": list(video_specs),
        "video_specs": {
            key: {"width": width, "height": height}
            for key, (width, height) in video_specs.items()
        },
        "summary": summary,
    }


def select_partial_plans(
    plans: Sequence[EpisodePlan],
    source_ids: set[str],
    episode_ids: set[str],
    source_keys: set[str],
    max_episodes: int | None,
) -> list[EpisodePlan]:
    normalized_source_keys = {
        key if key.startswith(f"{SEED}/") else f"{SEED}/{key}"
        for key in source_keys
    }
    selected = [
        plan
        for plan in plans
        if (not source_ids or plan.source_id in source_ids)
        and (not episode_ids or plan.source_episode_id in episode_ids)
        and (not normalized_source_keys or plan.source_key in normalized_source_keys)
    ]
    selected.sort(key=lambda item: (item.split, item.canonical_key))
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        selected = selected[:max_episodes]
    if not selected:
        raise ValueError("episode filters selected no episodes")
    return assign_root_indices(selected)


def convert_dataset(
    jsonl_archive: str | Path = DEFAULT_JSONL_ARCHIVE,
    frames_dir: str | Path = DEFAULT_FRAMES_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    work_dir: str | Path = DEFAULT_WORK_DIR,
    ffmpeg_path: str | Path = DEFAULT_FFMPEG,
    workers: int = 8,
    source_ids: set[str] | None = None,
    episode_ids: set[str] | None = None,
    source_keys: set[str] | None = None,
    max_episodes: int | None = None,
    allow_partial: bool = False,
    overwrite: bool = False,
    strict_full_inventory: bool = True,
) -> dict[str, Any]:
    jsonl_archive = Path(jsonl_archive).resolve()
    frames_dir = Path(frames_dir).resolve()
    output_dir = Path(output_dir).resolve()
    work_dir = Path(work_dir).resolve()
    ffmpeg_path = Path(ffmpeg_path).resolve()
    configure_temp_environment(work_dir)
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not jsonl_archive.is_file():
        raise FileNotFoundError(jsonl_archive)
    if not frames_dir.is_dir():
        raise FileNotFoundError(frames_dir)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {output_dir}")

    logging.info("Scanning source JSONL inventory: %s", jsonl_archive)
    inventory = scan_inventory(jsonl_archive)
    plans = assign_root_indices(split_inventory(inventory))
    if strict_full_inventory:
        validate_full_split(plans)

    filters_used = bool(source_ids or episode_ids or source_keys or max_episodes is not None)
    if filters_used:
        if not allow_partial:
            raise ValueError("episode filters require --allow-partial")
        plans = select_partial_plans(
            plans,
            source_ids or set(),
            episode_ids or set(),
            source_keys or set(),
            max_episodes,
        )
    elif allow_partial:
        raise ValueError("--allow-partial requires at least one episode filter")

    staging_dir = work_dir / f"{output_dir.name}.staging-{os.getpid()}"
    if staging_dir.exists():
        raise FileExistsError(f"staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    try:
        prepare_layout(staging_dir, plans)
        results = convert_episode_files(
            plans,
            jsonl_archive,
            frames_dir,
            staging_dir,
            ffmpeg_path,
            workers,
        )
        write_metadata(staging_dir, plans, results)
        summary = validate_output_dataset(staging_dir, decode_videos=False)
        if not filters_used and summary != EXPECTED_FULL_COUNTS:
            raise ValueError(f"full conversion summary mismatch: {summary}")
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(output_dir)
            shutil.rmtree(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_dir, output_dir)
    except BaseException:
        logging.error("Conversion failed; staging data retained at %s", staging_dir)
        raise
    return {
        "output_dir": str(output_dir),
        "partial": filters_used,
        "summary": summary,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert legacy or processed four-view OmTrackVLA data to LeRobot v2.1"
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        help=(
            "OmTrackVLA processed_tracking.v1 directory. When set, read direct four-view "
            "JPEG/JSONL files and write a single train root."
        ),
    )
    parser.add_argument("--jsonl-archive", type=Path, default=DEFAULT_JSONL_ARCHIVE)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument(
        "--source-key",
        action="append",
        default=[],
        help="Exact seed_101/source_id/episode_id key; repeatable",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--decode-videos", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    configure_temp_environment(args.work_dir)
    if args.validate_only:
        summary = validate_output_dataset(args.output_dir, decode_videos=args.decode_videos)
        print(_json_dump(summary, indent=2))
        return 0
    if args.processed_root is not None:
        result = convert_stage4_dataset(
            processed_root=args.processed_root,
            output_dir=args.output_dir,
            work_dir=args.work_dir,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        print(_json_dump(result, indent=2))
        return 0
    result = convert_dataset(
        jsonl_archive=args.jsonl_archive,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        ffmpeg_path=args.ffmpeg,
        workers=args.workers,
        source_ids=set(args.source_id),
        episode_ids=set(args.episode_id),
        source_keys=set(args.source_key),
        max_episodes=args.max_episodes,
        allow_partial=args.allow_partial,
        overwrite=args.overwrite,
    )
    print(_json_dump(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
