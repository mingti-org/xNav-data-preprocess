"""Read the prepared replay manifest and the selected annotation run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import av
import numpy as np

from unreal import (
    DEFAULT_CAMERA_KEYS,
    UnrealEpisode,
    intrinsic_4,
    intrinsic_matrix,
    load_json,
    load_jsonl,
    validate_fixed_extrinsics,
)

CAMERAS = tuple(DEFAULT_CAMERA_KEYS)
FPS = 10


@dataclass
class Selection:
    selected_ids: list[str]
    accepted: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    failed: list[dict[str, Any]]
    annotation: dict[str, Any]


def read_selection(
    input_root: Path, annotation_report: Path, episode_ids: list[str] | None = None
) -> Selection:
    report = load_json(annotation_report)
    if (
        report.get("schema") != "navigation_process_report"
        or report.get("version") != 1
        or report.get("source") != "human-replay"
        or report.get("task") != "vln-generate"
    ):
        raise ValueError("Expected a human-replay vln-generate report")
    annotation = report.get("annotation", {})
    if (
        not report.get("run_id")
        or not annotation.get("job_fingerprint")
        or not annotation.get("model")
        or annotation.get("prompt_version") != "nav_full_trajectory_v2"
        or annotation.get("sampling_fps") != FPS
    ):
        raise ValueError(
            "Report lacks the v2/10 FPS annotation contract; refresh it with nav-process report"
        )
    by_item = {item["item_id"]: item for item in report["items"]}
    if len(by_item) != len(report["items"]) or report["total"] != len(by_item):
        raise ValueError("Annotation report has duplicate items or inconsistent total")
    requested = {str(UUID(uid)) for uid in episode_ids} if episode_ids else None
    selected: list[dict] = []
    seen: set[str] = set()
    with (input_root / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            uid = str(UUID(row["episode_id"]))
            if requested is not None and uid not in requested:
                continue
            if uid in seen:
                raise ValueError(f"Duplicate selected UUID: {uid}")
            seen.add(uid)
            if (
                row.get("episode_relative_path") != f"episodes/{uid}"
                or row.get("status") != "prepared"
            ):
                raise ValueError(f"Invalid prepared manifest entry: {uid}")
            selected.append(row)
    if not selected or (requested is not None and seen != requested):
        raise ValueError(
            "Selected UUIDs are empty or missing from the prepared manifest"
        )

    result = Selection(
        selected_ids=[row["episode_id"] for row in selected],
        accepted=[],
        excluded=[],
        failed=[],
        annotation={"run_id": report["run_id"], **annotation},
    )
    for row in selected:
        uid = row["episode_id"]
        try:
            identity = hashlib.sha256(
                (row["source_revision"] + row["pipeline_fingerprint"]).encode()
            ).hexdigest()[:16]
            item = by_item.get(f"human-{uid}-{identity}")
            if item is None or item["status"] not in {
                "instruction_written",
                "filtered",
            }:
                status = item["status"] if item else "missing_from_annotation_run"
                raise ValueError(f"Annotation is not complete: {status}")
            directory = (input_root / row["episode_relative_path"]).resolve()
            if not directory.is_relative_to(input_root):
                raise ValueError("Episode path escapes input root")
            path = directory / "instruction.json"
            if (
                not item.get("output_path")
                or Path(item["output_path"]).resolve() != path
            ):
                raise ValueError(
                    "Annotation report points to a different instruction file"
                )
            payload = load_json(path)
            source, info = payload.get("source", {}), payload.get("annotation", {})
            if (
                payload.get("schema") != "navigation_instruction"
                or payload.get("version") != 2
            ):
                raise ValueError("Expected navigation_instruction version 2")
            for key in (
                "episode_id",
                "source_revision",
                "pipeline_fingerprint",
                "episode_relative_path",
            ):
                if source.get(key) != row[key]:
                    raise ValueError(f"Instruction source mismatch: {key}")
            for key in ("run_id", "job_fingerprint", "model", "prompt_version"):
                if info.get(key) != result.annotation[key]:
                    raise ValueError(f"Instruction annotation mismatch: {key}")
            if (
                info.get("annotation_sampling_fps") != FPS
                or info.get("source_frame_count") != row["frame_count"]
                or info.get("camera") != "front"
                or info.get("trajectory_usage") != "full"
            ):
                raise ValueError(
                    "Instruction does not describe this complete 10 FPS trajectory"
                )
            quality = payload.get("has_quality_issue")
            if type(quality) is not bool:
                raise ValueError("has_quality_issue must be a boolean")
            if quality:
                if (
                    item["status"] != "filtered"
                    or payload.get("vln") is not None
                    or not payload.get("quality_reason")
                ):
                    raise ValueError("Quality rejection disagrees with result/report")
                result.excluded.append(
                    {
                        "episode_id": uid,
                        "reason": "annotation_quality_issue",
                        "quality_reason": payload["quality_reason"],
                    }
                )
                continue
            instruction = (payload.get("vln") or {}).get("instruction")
            if (
                item["status"] != "instruction_written"
                or not isinstance(instruction, str)
                or not instruction.strip()
            ):
                raise ValueError("Missing completed VLN instruction")
            result.accepted.append(
                {**row, "instruction": instruction.strip(), "annotation": info}
            )
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result.failed.append(
                {"episode_id": uid, "stage": "annotation_handoff", "error": str(exc)}
            )
    return result


def validate_video(
    path: Path, count: int, image_size: tuple[int, int], *, decode: bool
) -> None:
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or container.streams.audio:
            raise ValueError(f"Expected one video stream without audio: {path}")
        stream = container.streams.video[0]
        if (
            float(stream.average_rate or 0) != FPS
            or stream.frames != count
            or (stream.height, stream.width) != image_size
        ):
            raise ValueError(f"Video count/FPS/size mismatch: {path}")
        if decode:
            decoded = 0
            for frame in container.decode(stream):
                if frame.pts is None or not np.isclose(
                    float(frame.pts * frame.time_base), decoded / FPS, atol=1e-5
                ):
                    raise ValueError(
                        f"Nonuniform output video timestamp at frame {decoded}: {path}"
                    )
                decoded += 1
            if decoded != count:
                raise ValueError(f"Decoded video count mismatch: {path}")


class HumanReplayEpisode(UnrealEpisode):
    def __init__(self, input_root: Path, record: dict, task_idx: int):
        directory = (input_root / record["episode_relative_path"]).resolve()
        meta = load_json(directory / "episode_meta.json")
        replay = load_json(directory / "replay.json")
        uid = record["episode_id"]
        count = record["frame_count"]
        if (
            meta.get("episode_id") != uid
            or meta.get("status") != "completed"
            or meta.get("sample_rate_hz") != FPS
            or meta.get("frame_count") != count
            or replay["source"]["episodeId"] != uid
            or replay["source"]["revision"] != record["source_revision"]
            or replay["pipeline"]["fingerprint"] != record["pipeline_fingerprint"]
            or replay["media"]["fps"] != FPS
            or replay["media"]["frameCount"] != count
            or replay["trajectory"]["frameCount"] != count
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
        ):
            raise ValueError(f"Replay identity/frame/FPS mismatch: {uid}")
        if set(meta.get("camera_names", [])) != set(CAMERAS):
            raise ValueError(f"Expected all four cameras: {uid}")
        image_size = (int(meta["capture_height"]), int(meta["capture_width"]))
        if min(image_size) <= 0:
            raise ValueError(f"Invalid capture size: {uid}")
        frames = load_jsonl(directory / "frames.jsonl")
        if len(frames) != count:
            raise ValueError(f"Frame/pose count mismatch: {uid}")
        for index, frame in enumerate(frames):
            if frame.get("frame_index") != index:
                raise ValueError(f"Non-contiguous frame_index: {uid}/{index}")
            for key in ("pose", *(f"camera_pose_{camera}" for camera in CAMERAS)):
                pose = np.asarray(frame.get(key), dtype=float)
                if pose.shape != (6,) or not np.isfinite(pose).all():
                    raise ValueError(f"Invalid {key}: {uid}/{index}")
            for camera in CAMERAS:
                key = f"K_{camera}"
                if key not in frame:
                    frame[key] = meta.get(key)
                k = np.asarray(frame[key], dtype=float)
                if (
                    k.shape != (9,)
                    or not np.isfinite(k).all()
                    or k[0] <= 0
                    or k[4] <= 0
                ):
                    raise ValueError(f"Invalid {key}: {uid}/{index}")
                if not np.allclose(k, frames[0][key], atol=1e-6, rtol=0):
                    raise ValueError(f"Changing {key}: {uid}/{index}")
        for camera in CAMERAS:
            validate_video(
                directory / f"rgb/{camera}.mp4", count, image_size, decode=False
            )
        extrinsics = validate_fixed_extrinsics(
            directory, frames, list(CAMERAS), 1e-4, 0.1
        )
        # The prepared replay owns these paths, including when original metadata has stale paths.
        meta["rgb_video_paths"] = {
            camera: str(directory / f"rgb/{camera}.mp4") for camera in CAMERAS
        }
        self.record, self.replay = record, replay
        self.task_meta = load_json(directory / "task_meta.json")
        super().__init__(
            directory,
            meta,
            frames,
            list(CAMERAS),
            record["instruction"],
            task_idx,
            [],
            extrinsics,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        result = {
            "source_episode_path": str(self.episode_dir),
            "source_episode_id": self.record["episode_id"],
            "source_revision": self.record["source_revision"],
            "pipeline_fingerprint": self.record["pipeline_fingerprint"],
            "source_archive_path": self.record.get("archive_path"),
            "scene_id": self.meta["scene_id"],
            "user_id": self.meta.get("username", ""),
            "original_episode_index": self.meta.get("episode_index", 0),
            "frame_count": len(self.frames),
            "fps": FPS,
            "capture_width": self.meta["capture_width"],
            "capture_height": self.meta["capture_height"],
            "camera_keys": list(CAMERAS),
            "task": self.task,
            "annotation": self.record["annotation"],
            "source_task": self.task_meta,
            "timestamp_semantics": "clean_replay_frame_index / 10; original timestamps retained in source frames.jsonl",
            "source_duration_seconds": self.replay["trajectory"].get("durationSec"),
            "playback_duration_seconds": len(self.frames) / FPS,
            "source_first_timestamp_us": self.frames[0].get("timestamp_us"),
            "source_last_timestamp_us": self.frames[-1].get("timestamp_us"),
        }
        for camera in CAMERAS:
            result[f"video.{camera}.K"] = intrinsic_4(self.frames[0], camera)
            result[f"video.{camera}.body_from_camera"] = self.body_from_camera[camera]
            result[f"K_{camera}"] = intrinsic_matrix(self.frames[0], camera)
            result[f"Extrinsic_{camera}"] = self.body_from_camera[camera]
        return result
