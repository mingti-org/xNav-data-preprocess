"""Convert replay-collected ObjectNav data to the Enactive LeRobot v2.1 contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pyarrow.parquet as pq

from utils.lerobot.lerobot_creater import LeRobotCreator
from utils.map2nav_vlnce.coordinates import habitat_poses_to_xnav

VIEWS = {
    "front": "video.front",
    "back": "video.rear",
    "left": "video.left",
    "right": "video.right",
}
ACTION_ALIASES = {
    0: {"STOP", "stop"},
    1: {"forward", "move_forward"},
    2: {"turn_left"},
    3: {"turn_right"},
}
EXPECTED_SOURCE_IDENTITIES = {
    "ovon": ("ovon", None),
    "hm3d_v2": ("hm3d_v2", None),
}


@dataclass(frozen=True)
class SourceSplit:
    partition: str
    root: Path


@dataclass(frozen=True)
class SourceEpisode:
    partition: str
    manifest: dict[str, Any]
    root: Path
    episode: dict[str, Any]
    source_key: str
    task: str
    length: int


class EpisodeIterator:
    def __init__(self, source: SourceEpisode, dataset_name: str):
        self.source = source
        self.dataset_name = dataset_name
        episode = source.episode
        self.metadata = {
            "source_episode_key": source.source_key,
            "source_partition": source.partition,
            "source_episode_id": episode["episode_id"],
            "source_trajectory_id": episode["trajectory_id"],
            "source_episode_dir_name": source.root.name,
            "dataset_name": dataset_name,
            "scene_key": episode["scene_key"],
            "object_category": episode["object_category"],
            "navigation_metrics": episode.get("metrics"),
        }
        self.video_sources = {
            output_key: str(source.root / f"{source_view}.mp4")
            for source_view, output_key in VIEWS.items()
        }

    def __iter__(self) -> Iterator[tuple[dict[str, Any], str]]:
        episode, steps = load_and_validate_episode(self.source, self.dataset_name)
        positions = np.asarray([step["position"] for step in steps], dtype=np.float64)
        rotations = np.asarray([step["rotation"] for step in steps], dtype=np.float64)
        states = habitat_poses_to_xnav(positions, rotations)
        _validate_source_videos(self.source, episode, len(steps))
        for index, step in enumerate(steps):
            yield {
                "observation.state": states[index],
                "action": np.asarray(
                    [step["discrete_action_to_next_id"]], dtype=np.int64
                ),
                "action_text": step["discrete_action_to_next"],
            }, self.source.task


def _validate_source_videos(
    source: SourceEpisode, episode: dict[str, Any], expected_frames: int
) -> None:
    expected_width = int(episode["video_width"])
    expected_height = int(episode["video_height"])
    expected_fps = float(episode["video_fps"])
    for source_view, output_key in VIEWS.items():
        path = source.root / f"{source_view}.mp4"
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"cannot open video: {output_key}")
            actual = (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(capture.get(cv2.CAP_PROP_FPS)),
                int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            )
            if actual[:2] != (expected_width, expected_height):
                raise ValueError(
                    f"video dimensions mismatch for {output_key}: "
                    f"{actual[0]}x{actual[1]}, expected {expected_width}x{expected_height}"
                )
            if abs(actual[2] - expected_fps) > 0.2:
                raise ValueError(
                    f"video FPS mismatch for {output_key}: {actual[2]}, "
                    f"expected {expected_fps}"
                )
            if actual[3] < expected_frames:
                raise ValueError(
                    f"video ended before frame {expected_frames}: "
                    f"{output_key} has {actual[3]} frames"
                )
            ok, _ = capture.read()
            if not ok:
                raise ValueError(f"cannot decode video: {output_key}")
        finally:
            capture.release()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def source_splits(root: Path) -> list[SourceSplit]:
    root = root.resolve()
    direct = root / "train"
    sharded: list[SourceSplit] = []
    for shard_root in sorted(path for path in root.glob("shard_*") if path.is_dir()):
        train = shard_root / "train"
        if train.is_dir():
            sharded.append(SourceSplit(shard_root.name, train.resolve()))
    hidden_shards = root / ".shards"
    if hidden_shards.is_dir():
        for shard_root in sorted(path for path in hidden_shards.iterdir() if path.is_dir()):
            train = shard_root / "train"
            if train.is_dir():
                sharded.append(
                    SourceSplit(f".shards/{shard_root.name}", train.resolve())
                )
    if direct.is_dir() and sharded:
        raise ValueError("input contains both direct train and sharded train directories")
    if direct.is_dir():
        return [SourceSplit("direct", direct.resolve())]
    if sharded:
        return sharded
    raise FileNotFoundError(
        f"no supported replay split found under {root}; expected train, shard_*/train, "
        "or .shards/*/train"
    )


def discover(
    root: Path, dataset_name: str
) -> tuple[list[SourceEpisode], list[dict[str, Any]], list[SourceSplit]]:
    splits = source_splits(root)
    sources: list[SourceEpisode] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_identity = EXPECTED_SOURCE_IDENTITIES[dataset_name]
    for source_split in splits:
        manifest_path = source_split.root / "manifest.jsonl"
        try:
            manifest_rows = _jsonl(manifest_path)
        except Exception as exc:
            errors.append(
                {
                    "source_key": source_split.partition,
                    "stage": "manifest_read",
                    "error": repr(exc),
                }
            )
            continue
        for row_number, manifest in enumerate(manifest_rows, 1):
            fallback_key = f"{source_split.partition}/manifest-row-{row_number}"
            source_key = fallback_key
            try:
                if not isinstance(manifest, dict):
                    raise ValueError("manifest row is not an object")
                relative = Path(str(manifest["episode_dir"]))
                if relative.is_absolute():
                    raise ValueError(f"absolute episode_dir: {relative}")
                episode_root = (source_split.root / relative).resolve()
                if source_split.root not in episode_root.parents:
                    raise ValueError(f"episode_dir escapes split: {relative}")
                directory_name = str(manifest.get("episode_dir_name") or relative.name)
                if directory_name != relative.name:
                    raise ValueError("episode_dir_name does not match episode_dir")
                source_key = f"{source_split.partition}/{directory_name}"
                if source_key in seen:
                    raise ValueError(f"duplicate source episode key: {source_key}")
                seen.add(source_key)
                episode = json.loads(
                    (episode_root / "episode.json").read_text(encoding="utf-8")
                )
                if (manifest.get("dataset"), manifest.get("role")) != expected_identity:
                    raise ValueError("manifest dataset/role does not match requested dataset")
                if (episode.get("dataset"), episode.get("role")) != expected_identity:
                    raise ValueError("episode dataset/role does not match requested dataset")
                for key in (
                    "dataset",
                    "role",
                    "split",
                    "episode_id",
                    "trajectory_id",
                    "scene_key",
                    "object_category",
                ):
                    if key in manifest and str(manifest[key]) != str(episode.get(key)):
                        raise ValueError(f"manifest/episode mismatch: {key}")
                instructions = episode.get("instructions")
                if not isinstance(instructions, list) or len(instructions) != 1:
                    raise ValueError("ObjectNav episode must contain exactly one instruction")
                task = instructions[0].get("instruction")
                if not isinstance(task, str) or not task.strip():
                    raise ValueError("missing instruction")
                length = int(manifest.get("num_steps", 0))
                if length <= 0:
                    raise ValueError("manifest num_steps must be positive")
                sources.append(
                    SourceEpisode(
                        partition=source_split.partition,
                        manifest=manifest,
                        root=episode_root,
                        episode=episode,
                        source_key=source_key,
                        task=task,
                        length=length,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "source_key": source_key,
                        "stage": "source_discovery",
                        "row": row_number,
                        "error": repr(exc),
                    }
                )
    return sorted(sources, key=lambda source: source.source_key), errors, splits


def load_and_validate_episode(
    source: SourceEpisode, dataset_name: str
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    root = source.root
    episode = json.loads((root / "episode.json").read_text(encoding="utf-8"))
    steps = tuple(_jsonl(root / "steps.jsonl"))
    required = ["trajectory.npz", *(f"{view}.mp4" for view in VIEWS)]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"missing files: {missing}")
    if any((root / name).stat().st_size <= 0 for name in required):
        raise ValueError("one or more required source files are empty")
    if (episode.get("dataset"), episode.get("role")) != EXPECTED_SOURCE_IDENTITIES[
        dataset_name
    ]:
        raise ValueError("episode dataset/role changed after discovery")
    count = len(steps)
    if (
        count != source.length
        or count != int(source.manifest.get("num_frames", -1))
        or count != int(episode.get("num_steps", -1))
        or count != int(episode.get("num_frames", -1))
    ):
        raise ValueError(
            f"length mismatch: steps={count}, manifest={source.length}, "
            f"episode={episode.get('num_steps')}"
        )
    for key in ("video_width", "video_height", "video_fps"):
        if float(episode.get(key, 0)) <= 0:
            raise ValueError(f"invalid {key}")
    with np.load(root / "trajectory.npz", allow_pickle=False) as trajectory:
        required_arrays = {
            "positions",
            "rotations",
            "discrete_action_to_next_ids",
            "video_frame_indices",
        }
        if not required_arrays <= set(trajectory.files):
            raise ValueError(
                f"missing npz arrays: {sorted(required_arrays - set(trajectory.files))}"
            )
        if any(trajectory[name].shape[0] != count for name in required_arrays):
            raise ValueError("NPZ length mismatch")
        for index, step in enumerate(steps):
            if int(step.get("step_index", -1)) != index:
                raise ValueError(f"non-contiguous step index at {index}")
            if int(step.get("video_frame_index", -1)) != index:
                raise ValueError(f"video frame index mismatch at {index}")
            action_id = int(step.get("discrete_action_to_next_id", -1))
            action_text = step.get("discrete_action_to_next")
            if action_id not in ACTION_ALIASES or action_text not in ACTION_ALIASES[action_id]:
                raise ValueError(f"unknown/inconsistent action at {index}")
            if not np.allclose(trajectory["positions"][index], step["position"], atol=1e-5):
                raise ValueError(f"position conflict at {index}")
            if not np.allclose(trajectory["rotations"][index], step["rotation"], atol=1e-5):
                raise ValueError(f"rotation conflict at {index}")
            if (
                int(trajectory["discrete_action_to_next_ids"][index]) != action_id
                or int(trajectory["video_frame_indices"][index]) != index
            ):
                raise ValueError(f"NPZ action/index conflict at {index}")
    if not steps:
        raise ValueError("empty steps")
    if int(steps[-1]["discrete_action_to_next_id"]) != 0:
        raise ValueError("last frame must contain STOP=0")
    return episode, steps


def _fingerprint(splits: list[SourceSplit]) -> str:
    digest = hashlib.sha256()
    for source_split in splits:
        path = source_split.root / "manifest.jsonl"
        digest.update(source_split.partition.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _recorded_error_count(splits: list[SourceSplit]) -> int:
    return sum(len(_jsonl(source_split.root / "errors.jsonl")) for source_split in splits)


def _source_manifest_rows(sources: list[SourceEpisode]) -> list[dict[str, Any]]:
    return [
        {
            "source_episode_key": source.source_key,
            "source_partition": source.partition,
            "episode_dir": str(source.root),
            "num_frames": source.length,
            "task": source.task,
        }
        for source in sources
    ]


def _video_contract(sources: list[SourceEpisode]) -> tuple[int, int, int]:
    contracts = {
        (
            int(source.episode.get("video_width", 0)),
            int(source.episode.get("video_height", 0)),
            int(source.episode.get("video_fps", 0)),
        )
        for source in sources
    }
    if len(contracts) != 1:
        raise ValueError(f"source episodes do not share one video contract: {contracts}")
    width, height, fps = contracts.pop()
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError(f"invalid video contract: {width}x{height}@{fps}")
    return width, height, fps


def _validate_output(
    root: Path,
    expected: list[SourceEpisode],
    task_indices: dict[str, int],
    *,
    width: int,
    height: int,
    fps: int,
) -> dict[str, int]:
    meta = root / "meta"
    required = [
        meta / "info.json",
        meta / "tasks.jsonl",
        meta / "episodes.jsonl",
        meta / "episodes_stats.jsonl",
        meta / "episodes_extras.jsonl",
    ]
    if any(not path.is_file() for path in required):
        raise ValueError("missing LeRobot metadata output")
    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    tasks = _jsonl(meta / "tasks.jsonl")
    actual_tasks = {row["task"]: int(row["task_index"]) for row in tasks}
    if actual_tasks != task_indices:
        raise ValueError("task index mapping mismatch")
    episodes = _jsonl(meta / "episodes.jsonl")
    extras = _jsonl(meta / "episodes_extras.jsonl")
    stats = _jsonl(meta / "episodes_stats.jsonl")
    if not (len(episodes) == len(extras) == len(stats) == len(expected)):
        raise ValueError("episode metadata count mismatch")
    episodes_by_index = {int(row["episode_index"]): row for row in episodes}
    extras_by_index = {int(row["episode_index"]): row for row in extras}
    stats_indices = {int(row["episode_index"]) for row in stats}
    expected_indices = set(range(len(expected)))
    if (
        set(episodes_by_index) != expected_indices
        or set(extras_by_index) != expected_indices
        or stats_indices != expected_indices
    ):
        raise ValueError("episode indices are not contiguous across metadata files")
    sources_by_key = {source.source_key: source for source in expected}
    if {row.get("source_episode_key") for row in extras} != set(sources_by_key):
        raise ValueError("source provenance mismatch")

    total_frames = 0
    for episode_index in range(len(expected)):
        extra = extras_by_index[episode_index]
        source = sources_by_key[extra["source_episode_key"]]
        episode_meta = episodes_by_index[episode_index]
        if int(episode_meta["length"]) != source.length:
            raise ValueError(f"episode length mismatch for {source.source_key}")
        if episode_meta.get("tasks") != [source.task]:
            raise ValueError(f"episode task mismatch for {source.source_key}")
        parquet_path = (
            root
            / "data"
            / f"chunk-{episode_index // 1000:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        table = pq.read_table(
            parquet_path,
            columns=[
                "observation.state",
                "action",
                "action_text",
                "frame_index",
                "episode_index",
                "task_index",
            ],
        )
        if table.num_rows != source.length:
            raise ValueError(f"parquet row count mismatch for {source.source_key}")
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        if states.shape != (source.length, 7) or not np.all(np.isfinite(states)):
            raise ValueError(f"invalid state output for {source.source_key}")
        if not np.allclose(
            states[0],
            np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            atol=1e-5,
        ):
            raise ValueError(f"state origin mismatch for {source.source_key}")
        if table["frame_index"].to_pylist() != list(range(source.length)):
            raise ValueError(f"frame indices mismatch for {source.source_key}")
        if set(table["episode_index"].to_pylist()) != {episode_index}:
            raise ValueError(f"parquet episode index mismatch for {source.source_key}")
        if set(table["task_index"].to_pylist()) != {task_indices[source.task]}:
            raise ValueError(f"parquet task index mismatch for {source.source_key}")
        if int(table["action"].to_pylist()[-1][0]) != 0:
            raise ValueError(f"parquet terminal action mismatch for {source.source_key}")
        if table["action_text"].to_pylist()[-1] not in ACTION_ALIASES[0]:
            raise ValueError(f"parquet terminal action text mismatch for {source.source_key}")
        for video_key in VIEWS.values():
            video_path = (
                root
                / "videos"
                / f"chunk-{episode_index // 1000:03d}"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            capture = cv2.VideoCapture(str(video_path))
            try:
                if not capture.isOpened():
                    raise ValueError(f"cannot open output video: {video_path}")
                actual = (
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(round(capture.get(cv2.CAP_PROP_FPS))),
                    int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                )
                if actual[:3] != (width, height, fps) or actual[3] < source.length:
                    raise ValueError(
                        f"output video metadata mismatch for {video_path}: {actual}"
                    )
                ok, _ = capture.read()
                if not ok:
                    raise ValueError(f"cannot decode output video: {video_path}")
            finally:
                capture.release()
            source_view = next(
                view for view, output_key in VIEWS.items() if output_key == video_key
            )
            source_video = source.root / f"{source_view}.mp4"
            if video_path.stat().st_size != source_video.stat().st_size:
                raise ValueError(f"output video size differs from source: {video_path}")
        total_frames += source.length
    if (
        int(info.get("total_episodes", -1)) != len(expected)
        or int(info.get("total_frames", -1)) != total_frames
        or int(info.get("total_tasks", -1)) != len(task_indices)
        or int(info.get("total_videos", -1)) != len(expected) * len(VIEWS)
    ):
        raise ValueError("info.json totals do not match validated outputs")
    return {"validated_episodes": len(expected), "validated_frames": total_frames}


def _failure_report(
    staging: Path,
    *,
    dataset_name: str,
    source_count: int,
    recorded_errors: int,
    errors: list[dict[str, Any]],
) -> None:
    _write_jsonl(staging / "errors.jsonl", errors)
    _write_json(
        staging / "conversion_report.json",
        {
            "dataset_name": dataset_name,
            "publishable": False,
            "source_success_manifest_rows": source_count,
            "source_recorded_errors": recorded_errors,
            "converted_episodes": 0,
            "conversion_errors": len(errors),
            "skipped_episodes": 0,
        },
    )


def convert(
    input_root: Path,
    output_root: Path,
    dataset_name: str,
    *,
    workers: int = 1,
    overwrite: bool = False,
) -> None:
    if output_root.exists() and not overwrite:
        raise FileExistsError(output_root)
    staging = Path(str(output_root) + ".staging")
    if staging.exists() and not overwrite:
        raise FileExistsError(staging)
    if overwrite:
        if output_root.exists():
            shutil.rmtree(output_root)
        if staging.exists():
            shutil.rmtree(staging)

    sources, discovery_errors, splits = discover(input_root, dataset_name)
    recorded_errors = _recorded_error_count(splits)
    if not sources:
        discovery_errors.append(
            {
                "source_key": None,
                "stage": "source_discovery",
                "error": "no valid source episodes discovered",
            }
        )
    if discovery_errors:
        staging.mkdir(parents=True, exist_ok=True)
        _failure_report(
            staging,
            dataset_name=dataset_name,
            source_count=len(sources),
            recorded_errors=recorded_errors,
            errors=discovery_errors,
        )
        raise ValueError(
            f"source discovery failed with {len(discovery_errors)} error(s); "
            f"report kept at {staging}"
        )

    width, height, fps = _video_contract(sources)
    staging.mkdir(parents=True, exist_ok=False)
    _write_jsonl(staging / "source_manifest.jsonl", _source_manifest_rows(sources))
    tasks = sorted({source.task for source in sources})
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": {"axes": ["x", "y", "z", "qx", "qy", "qz", "qw"]},
        },
        **{
            name: {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
            }
            for name in VIEWS.values()
        },
        "action": {"dtype": "int64", "shape": [1], "names": None},
        "action_text": {"dtype": "string", "shape": [1], "names": None},
    }
    creator = LeRobotCreator(
        str(staging),
        robot_type="habitat_objectnav",
        fps=fps,
        features=features,
        num_workers=workers,
        num_video_encoders=0,
        has_extras=True,
    )
    task_indices: dict[str, int] = {}
    try:
        for task in tasks:
            task_indices[task] = int(creator.add_task(task))
        for source in sources:
            creator.submit_episode(EpisodeIterator(source, dataset_name))
        creator.wait()
        validation = _validate_output(
            staging,
            sources,
            task_indices,
            width=width,
            height=height,
            fps=fps,
        )
    except Exception as exc:
        if not creator.finished:
            try:
                creator.wait()
            except Exception:
                pass
        errors = creator.errors or [
            {
                "source_key": None,
                "stage": "output_validation",
                "error": repr(exc),
            }
        ]
        _failure_report(
            staging,
            dataset_name=dataset_name,
            source_count=len(sources),
            recorded_errors=recorded_errors,
            errors=errors,
        )
        raise

    _write_json(staging / "validation_report.json", validation)
    _write_json(
        staging / "conversion_report.json",
        {
            "dataset_name": dataset_name,
            "publishable": True,
            "coordinate_frame": "xnav_episode_start_relative",
            "copy_mode": "copy2",
            "source_layouts": [source_split.partition for source_split in splits],
            "source_success_manifest_rows": len(sources),
            "source_recorded_errors": recorded_errors,
            "converted_episodes": len(sources),
            "conversion_errors": 0,
            "skipped_episodes": 0,
            "input_fingerprint": _fingerprint(splits),
            "task_indices": task_indices,
        },
    )
    os.replace(staging, output_root)


def audit(input_root: Path, output_root: Path, dataset_name: str) -> None:
    sources, errors, splits = discover(input_root, dataset_name)
    for source in sources:
        try:
            load_and_validate_episode(source, dataset_name)
        except Exception as exc:
            errors.append(
                {
                    "source_key": source.source_key,
                    "stage": "source_validation",
                    "error": repr(exc),
                }
            )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "source_manifest.jsonl", _source_manifest_rows(sources))
    _write_jsonl(output_root / "errors.jsonl", errors)
    _write_json(
        output_root / "conversion_report.json",
        {
            "dataset_name": dataset_name,
            "publishable": False,
            "audit_only": True,
            "source_success_manifest_rows": len(sources),
            "source_recorded_errors": _recorded_error_count(splits),
            "conversion_errors": len(errors),
            "skipped_episodes": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-name", choices=sorted(EXPECTED_SOURCE_IDENTITIES), required=True
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be positive")
    if args.audit_only:
        audit(args.input_root, args.output_root, args.dataset_name)
        return
    convert(
        args.input_root,
        args.output_root,
        args.dataset_name,
        workers=args.num_workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
