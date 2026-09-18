"""Convert HM3D OVON replay output to the Enactive LeRobot v2.1 contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Iterator

import cv2
import numpy as np

from utils.lerobot.lerobot_creater import LeRobotCreator

VIEWS = {"front": "video.front", "back": "video.rear", "left": "video.left", "right": "video.right"}
ACTION_NAMES = {0: "STOP", 1: "forward", 2: "turn_left", 3: "turn_right"}


@dataclass(frozen=True)
class SourceEpisode:
    shard: str
    manifest: dict
    root: Path
    episode: dict
    steps: tuple[dict, ...]
    source_key: str


class EpisodeIterator:
    def __init__(self, source: SourceEpisode, task: str):
        self.source = source
        self.task = task
        self.metadata = {
            "source_episode_key": source.source_key,
            "source_shard": source.shard,
            "source_episode_id": source.episode["episode_id"],
            "scene_key": source.episode["scene_key"],
            "object_category": source.episode["object_category"],
        }

    def __iter__(self) -> Iterator[tuple[dict, str]]:
        captures = {VIEWS[v]: cv2.VideoCapture(str(self.source.root / f"{v}.mp4")) for v in VIEWS}
        try:
            for i, step in enumerate(self.source.steps):
                frame: dict = {}
                for key, cap in captures.items():
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError(f"video ended before frame {i}: {key}")
                    if image.shape[:2] != (480, 640):
                        raise ValueError(f"unexpected video shape for {key}: {image.shape}")
                    frame[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame["observation.state"] = np.asarray(step["position"] + step["rotation"], dtype=np.float32)
                frame["action"] = np.asarray([step["discrete_action_to_next_id"]], dtype=np.int64)
                frame["action_text"] = step["discrete_action_to_next"]
                yield frame, self.task
        finally:
            for cap in captures.values():
                cap.release()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def discover(root: Path) -> tuple[list[SourceEpisode], list[dict]]:
    sources: list[SourceEpisode] = []
    errors: list[dict] = []
    seen: set[str] = set()
    for shard_root in sorted(root.glob("shard_*")):
        train = (shard_root / "train").resolve()
        manifest_path = train / "manifest.jsonl"
        for row_no, manifest in enumerate(_jsonl(manifest_path), 1):
            rel = Path(manifest["episode_dir"])
            if rel.is_absolute():
                raise ValueError(f"absolute episode_dir: {manifest['episode_dir']}")
            episode_root = (train / rel).resolve()
            if train not in episode_root.parents:
                raise ValueError(f"episode_dir escapes shard: {manifest['episode_dir']}")
            key = f"{shard_root.name}/{manifest['episode_dir_name']}"
            if key in seen:
                raise ValueError(f"duplicate source episode key: {key}")
            seen.add(key)
            try:
                episode = json.loads((episode_root / "episode.json").read_text())
                steps = tuple(_jsonl(episode_root / "steps.jsonl"))
                validate_episode(episode_root, manifest, episode, steps)
                sources.append(SourceEpisode(shard_root.name, manifest, episode_root, episode, steps, key))
            except Exception as exc:
                errors.append({"source_key": key, "stage": "source_validation", "error": repr(exc), "row": row_no})
    return sorted(sources, key=lambda x: x.source_key), errors


def validate_episode(root: Path, manifest: dict, episode: dict, steps: tuple[dict, ...]) -> None:
    required = ["episode.json", "steps.jsonl", "trajectory.npz", *(f"{v}.mp4" for v in VIEWS)]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"missing files: {missing}")
    n = len(steps)
    if not episode.get("instructions") or not episode["instructions"][0].get("instruction"):
        raise ValueError("missing instruction")
    for key in ("episode_id", "scene_key", "trajectory_id"):
        if key in manifest and str(manifest[key]) != str(episode.get(key)):
            raise ValueError(f"manifest/episode mismatch: {key}")
    if n != manifest["num_steps"] or n != episode["num_steps"] or n != episode["num_frames"]:
        raise ValueError(f"length mismatch: steps={n}, manifest={manifest['num_steps']}, episode={episode['num_steps']}")
    npz = np.load(root / "trajectory.npz", allow_pickle=False)
    required_arrays = {"positions", "rotations", "discrete_action_to_next_ids", "video_frame_indices"}
    if not required_arrays <= set(npz.files):
        raise ValueError(f"missing npz arrays: {sorted(required_arrays - set(npz.files))}")
    if any(npz[name].shape[0] != n for name in required_arrays):
        raise ValueError("NPZ length mismatch")
    for i, step in enumerate(steps):
        if step["step_index"] != i or step["video_frame_index"] != i:
            raise ValueError(f"non-contiguous frame index at {i}")
        action_id = int(step["discrete_action_to_next_id"])
        if action_id not in ACTION_NAMES or step["discrete_action_to_next"] != ACTION_NAMES[action_id]:
            raise ValueError(f"unknown/inconsistent action at {i}")
        if not np.allclose(npz["positions"][i], step["position"], atol=1e-5):
            raise ValueError(f"position conflict at {i}")
        if not np.allclose(npz["rotations"][i], step["rotation"], atol=1e-5):
            raise ValueError(f"rotation conflict at {i}")
        if int(npz["discrete_action_to_next_ids"][i]) != action_id or int(npz["video_frame_indices"][i]) != i:
            raise ValueError(f"NPZ action/index conflict at {i}")
    if not steps:
        raise ValueError("empty steps")
    if steps[-1]["discrete_action_to_next_id"] != 0:
        raise ValueError("last frame must contain STOP=0")


def _frames(path: Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _fingerprint(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.glob("shard_*/train/manifest.jsonl")):
        h.update(p.read_bytes())
    return h.hexdigest()


def _validate_output(root: Path, expected: list[SourceEpisode], task_indices: dict[str, int]) -> None:
    meta = root / "meta"
    required = [meta / "info.json", meta / "tasks.jsonl", meta / "episodes.jsonl", meta / "episodes_extras.jsonl"]
    if any(not p.is_file() for p in required):
        raise ValueError("missing LeRobot metadata output")
    tasks = _jsonl(meta / "tasks.jsonl")
    actual_tasks = {row["task"]: int(row["task_index"]) for row in tasks}
    if actual_tasks != task_indices:
        raise ValueError("task index mapping mismatch")
    episodes = _jsonl(meta / "episodes.jsonl")
    extras = _jsonl(meta / "episodes_extras.jsonl")
    if len(episodes) != len(expected) or len(extras) != len(expected):
        raise ValueError("episode metadata count mismatch")
    source_keys = {row.get("source_episode_key") for row in extras}
    if source_keys != {s.source_key for s in expected}:
        raise ValueError("source provenance mismatch")
    if sum(int(row["length"]) for row in episodes) != sum(len(s.steps) for s in expected):
        raise ValueError("output frame count mismatch")


def convert(input_root: Path, output_root: Path, *, workers: int = 1, overwrite: bool = False) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    staging = Path(str(output_root) + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    sources, source_errors = discover(input_root)
    if source_errors:
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "errors.jsonl").write_text("".join(json.dumps(e) + "\n" for e in source_errors))
        (staging / "conversion_report.json").write_text(json.dumps({"publishable": False, "source_recorded_errors": 0, "converted_episodes": 0, "conversion_errors": len(source_errors), "skipped_episodes": 0}, indent=2) + "\n")
        raise ValueError(f"source validation failed for {len(source_errors)} episodes; report kept at {staging}")
    tasks = sorted({s.episode["instructions"][0]["instruction"] for s in sources})
    features = {
        "observation.state": {"dtype": "float32", "shape": [7], "names": {"axes": ["x", "y", "z", "qx", "qy", "qz", "qw"]}},
        **{name: {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"]} for name in VIEWS.values()},
        "action": {"dtype": "int64", "shape": [1], "names": None},
        "action_text": {"dtype": "string", "shape": [1], "names": None},
    }
    creator = LeRobotCreator(str(staging), fps=10, features=features, num_workers=workers, num_video_encoders=min(workers, 16), has_extras=True)
    task_indices: dict[str, int] = {}
    try:
        for task in tasks:
            task_indices[task] = creator.add_task(task)
        for source in sources:
            task = source.episode["instructions"][0]["instruction"]
            creator.submit_episode(EpisodeIterator(source, task))
        creator.wait()
    except Exception:
        try:
            creator.wait()
        except Exception:
            pass
        (staging / "conversion_report.json").write_text(json.dumps({"publishable": False, "source_recorded_errors": 0, "converted_episodes": 0, "conversion_errors": 1, "skipped_episodes": 0}, indent=2) + "\n")
        raise
    _validate_output(staging, sources, task_indices)
    recorded_errors = sum(len(_jsonl(p / "errors.jsonl")) for p in input_root.glob("shard_*/train"))
    (staging / "conversion_report.json").write_text(json.dumps({"publishable": True, "source_recorded_errors": recorded_errors, "converted_episodes": len(sources), "conversion_errors": 0, "skipped_episodes": 0, "input_fingerprint": _fingerprint(input_root), "task_indices": task_indices}, indent=2) + "\n")
    staging.rename(output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be positive")
    sources, errors = discover(args.input_root)
    if args.audit_only:
        audit_dir = args.output_root
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "source_manifest.jsonl").write_text("".join(json.dumps({"source_key": s.source_key, "episode_dir": str(s.root), "num_frames": len(s.steps)}) + "\n" for s in sources))
        (audit_dir / "errors.jsonl").write_text("".join(json.dumps(e) + "\n" for e in errors))
        (audit_dir / "conversion_report.json").write_text(json.dumps({"publishable": not errors, "source_success_manifest_rows": len(sources), "conversion_errors": len(errors), "skipped_episodes": 0}, indent=2) + "\n")
        return
    convert(args.input_root, args.output_root, workers=args.num_workers, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
