"""Write and validate one complete human replay training dataset."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from unreal import (
    ACTION_KEY,
    STATE_KEY,
    TASK_DESCRIPTION_KEY,
    build_features,
    load_json,
    load_jsonl_dicts,
    utc_now_iso,
    validate_lerobot_dataset,
    write_json,
)
from utils.human_replay.source import (
    CAMERAS,
    FPS,
    HumanReplayEpisode,
    read_selection,
    validate_video,
)

REPORT_PATH = "meta/human_replay_conversion_report.json"


def write_modality(root: Path) -> None:
    write_json(
        root / "meta/modality.json",
        {
            "state": {"drone": {"start": 0, "end": 7, "original_key": STATE_KEY}},
            "action": {
                "state": {
                    "start": 0,
                    "end": 7,
                    "absolute": True,
                    "original_key": ACTION_KEY,
                }
            },
            "video": {
                camera: {"original_key": f"video.{camera}"} for camera in CAMERAS
            },
            "annotation": {
                TASK_DESCRIPTION_KEY: {"original_key": TASK_DESCRIPTION_KEY}
            },
        },
    )


def validate_output(
    root: Path, records: list[dict], image_size: tuple[int, int]
) -> list[dict]:
    """Check actual files, not submitted work; fix the writer's per-episode index to global index."""
    info = load_json(root / "meta/info.json")
    episodes = load_jsonl_dicts(root / "meta/episodes.jsonl")
    extras = load_jsonl_dicts(root / "meta/episodes_extras.jsonl")
    stats = load_jsonl_dicts(root / "meta/episodes_stats.jsonl")
    tasks = load_jsonl_dicts(root / "meta/tasks.jsonl")
    expected = {row["episode_id"]: row for row in records}
    indices = set(range(len(records)))
    for name, rows in (("episodes", episodes), ("extras", extras), ("stats", stats)):
        if (
            len(rows) != len(records)
            or {row["episode_index"] for row in rows} != indices
        ):
            raise ValueError(f"Incomplete or duplicate {name} metadata")
    actual_ids = [row.get("source_episode_id") for row in extras]
    if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != set(expected):
        raise ValueError("Written source UUIDs do not match accepted annotations")
    frame_count = sum(record["frame_count"] for record in records)
    video_keys = {f"video.{cam}" for cam in CAMERAS}
    if (
        info["fps"] != FPS
        or info["total_episodes"] != len(records)
        or info["total_frames"] != frame_count
        or info["total_videos"] != len(records) * len(CAMERAS)
        or {k for k, v in info["features"].items() if v["dtype"] == "video"}
        != video_keys
        or info["total_tasks"] != len(tasks)
    ):
        raise ValueError("Dataset totals/FPS/video features disagree with selection")
    task_map = {row["task_index"]: row["task"] for row in tasks}
    if len(task_map) != len(tasks):
        raise ValueError("Duplicate task indices")
    by_episode = {row["episode_index"]: row for row in episodes}
    by_stats = {row["episode_index"]: row["stats"] for row in stats}
    successful = []
    frame_offset = 0
    for extra in sorted(extras, key=lambda row: row["episode_index"]):
        index = extra["episode_index"]
        record = expected[extra["source_episode_id"]]
        n = record["frame_count"]
        episode = by_episode[index]
        if (
            episode["length"] != n
            or extra["frame_count"] != n
            or episode["tasks"] != [record["instruction"]]
        ):
            raise ValueError(f"Episode length/task mismatch: {index}")
        if extra["annotation"] != record["annotation"]:
            raise ValueError(f"Episode annotation provenance mismatch: {index}")
        path = root / info["data_path"].format(
            episode_chunk=index // info["chunks_size"], episode_index=index
        )
        table = pq.read_table(path)
        if table.num_rows != n:
            raise ValueError(f"Parquet row count mismatch: {index}")
        state = np.asarray(table[STATE_KEY].to_pylist())
        action = np.asarray(table[ACTION_KEY].to_pylist())
        if (
            state.shape != (n, 7)
            or not np.isfinite(state).all()
            or not np.array_equal(state, action)
            or not np.allclose(np.linalg.norm(state[:, 3:], axis=1), 1, atol=1e-5)
            or not np.allclose(state[0], [0, 0, 0, 0, 0, 0, 1], atol=1e-5)
        ):
            raise ValueError(f"Invalid relative pose/state/action: {index}")
        if (
            table["frame_index"].to_pylist() != list(range(n))
            or table["episode_index"].to_pylist() != [index] * n
            or not np.allclose(
                table["timestamp"].to_numpy(), np.arange(n) / FPS, atol=1e-5
            )
        ):
            raise ValueError(f"Parquet frame/time alignment mismatch: {index}")
        task_indices = table["task_index"].to_pylist()
        if any(task_map.get(t) != record["instruction"] for t in task_indices):
            raise ValueError(f"Parquet task mapping mismatch: {index}")
        if (
            np.asarray(table[TASK_DESCRIPTION_KEY].to_pylist()).reshape(-1).tolist()
            != task_indices
        ):
            raise ValueError(f"Instruction/task index mismatch: {index}")
        # The shared writer deliberately excludes annotation indices from numeric stats.
        required_stats = video_keys | {STATE_KEY, ACTION_KEY}
        if not required_stats.issubset(by_stats[index]):
            raise ValueError(f"Missing feature statistics: {index}")
        for feature in required_stats:
            if not all(
                np.isfinite(np.asarray(value, dtype=float)).all()
                for value in by_stats[index][feature].values()
            ):
                raise ValueError(f"Nonfinite feature statistics: {index}/{feature}")
        for camera in CAMERAS:
            video = root / info["video_path"].format(
                episode_chunk=index // info["chunks_size"],
                episode_index=index,
                video_key=f"video.{camera}",
            )
            validate_video(video, n, image_size, decode=True)
        # Standard LeRobot index spans episodes. This changes neither frame_index nor poses.
        global_index = pa.array(
            np.arange(frame_offset, frame_offset + n, dtype=np.int64)
        )
        if not table["index"].combine_chunks().equals(global_index):
            table = table.set_column(
                table.schema.get_field_index("index"), "index", global_index
            )
            temporary = path.with_suffix(".parquet.tmp")
            pq.write_table(table, temporary)
            os.replace(temporary, path)
        successful.append(
            {
                "source_episode_id": record["episode_id"],
                "episode_index": index,
                "frame_count": n,
                "source_revision": record["source_revision"],
            }
        )
        frame_offset += n
    validate_lerobot_dataset(root.name, root)
    return successful


def _abort_writer(creator) -> None:
    """Release only this conversion's workers when the caller cannot finish submission."""
    processes = [*creator.workers, *creator.encoders, creator.meta_process]
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
    # No reader remains for queued payloads after abort; do not block at interpreter exit.
    for queue in (
        creator.task_queue,
        creator.video_queue,
        creator.meta_req_queue,
        *creator.reply_queues,
    ):
        queue.cancel_join_thread()
        queue.close()


def convert(
    input_root: Path,
    annotation_report: Path,
    output_root: Path,
    *,
    episode_ids: list[str] | None = None,
    num_workers: int = 1,
) -> dict:
    input_root, annotation_report, output_root = (
        input_root.resolve(),
        annotation_report.resolve(),
        output_root.resolve(),
    )
    if (
        output_root == input_root
        or output_root.is_relative_to(input_root)
        or input_root.is_relative_to(output_root)
    ):
        raise ValueError("Output must be separate from the prepared input tree")
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    staging = output_root.with_name(output_root.name + ".staging")
    if output_root.exists() or staging.exists():
        raise FileExistsError(
            f"Refusing to overwrite output or staging: {output_root}, {staging}"
        )
    staging.mkdir(parents=True)
    report = {
        "schema": "human_replay_conversion_report",
        "version": 1,
        "status": "preparing",
        "started_at": utc_now_iso(),
        "input_root": str(input_root),
        "annotation_report": str(annotation_report),
        "output_root": str(output_root),
        "fps": FPS,
        "camera_keys": list(CAMERAS),
        "selection": "explicit_uuids" if episode_ids else "all_prepared",
        "selected_episode_ids": [],
        "num_selected": 0,
        "num_successful": 0,
        "num_excluded": 0,
        "num_failed": 0,
        "excluded_episodes": [],
        "failed_episodes": [],
        "successful_episodes": [],
    }
    creator = None
    started = time.monotonic()
    try:
        selection = read_selection(input_root, annotation_report, episode_ids)
        report.update(
            annotation=selection.annotation,
            selected_episode_ids=selection.selected_ids,
            num_selected=len(selection.selected_ids),
            num_excluded=len(selection.excluded),
            num_failed=len(selection.failed),
            excluded_episodes=selection.excluded,
            failed_episodes=selection.failed,
        )
        if selection.failed:
            raise ValueError(
                f"Unresolved annotations for {len(selection.failed)} selected episodes"
            )
        if not selection.accepted:
            raise ValueError("No valid VLN episodes after annotation quality filtering")
        from utils.lerobot.lerobot_creater import LeRobotCreator

        image_size = None
        report["status"] = "converting"
        for index, record in enumerate(selection.accepted, 1):
            try:
                # Only the current episode and bounded writer queue retain frames in memory.
                episode = HumanReplayEpisode(input_root, record, task_idx=0)
                current_size = (
                    episode.meta["capture_height"],
                    episode.meta["capture_width"],
                )
                if creator is None:
                    image_size = current_size
                    creator = LeRobotCreator(
                        root=str(staging),
                        robot_type="go2",
                        fps=FPS,
                        features=build_features(image_size, CAMERAS),
                        num_workers=num_workers,
                        num_video_encoders=num_workers,
                        codec="h264",
                        pix_fmt="yuv420p",
                        has_extras=True,
                    )
                if current_size != image_size:
                    raise ValueError(
                        "Mixed image sizes require separate output datasets"
                    )
                episode.task_idx = creator.add_task(episode.task)
                creator.submit_episode(episode)
            except Exception as exc:
                report["failed_episodes"].append(
                    {
                        "episode_id": record["episode_id"],
                        "stage": "episode_prepare",
                        "error": str(exc),
                    }
                )
                raise
            report["num_submitted"] = index
            write_json(staging / REPORT_PATH, report)
            logging.info(
                "转换已提交=%d/%d UUID=%s",
                index,
                len(selection.accepted),
                record["episode_id"],
            )
        creator.wait()
        creator = None
        report["status"] = "validating"
        write_json(staging / REPORT_PATH, report)
        write_modality(staging)
        report["successful_episodes"] = validate_output(
            staging, selection.accepted, image_size
        )
        report.update(
            status="completed",
            num_successful=len(report["successful_episodes"]),
            total_frames=sum(
                row["frame_count"] for row in report["successful_episodes"]
            ),
            completed_at=utc_now_iso(),
            elapsed_seconds=round(time.monotonic() - started, 2),
        )
        write_json(staging / REPORT_PATH, report)
        if output_root.exists():
            raise FileExistsError(f"Output appeared during conversion: {output_root}")
        staging.rename(output_root)
        logging.info(
            "转换完成=%d 帧数=%d 输出=%s",
            report["num_successful"],
            report["total_frames"],
            output_root,
        )
        return report
    except BaseException as exc:
        if creator is not None:
            _abort_writer(creator)
        report.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            num_failed=len(report["failed_episodes"]),
            completed_at=utc_now_iso(),
        )
        write_json(staging / REPORT_PATH, report)
        raise
