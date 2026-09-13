from __future__ import annotations

import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import pytest

from human_replay import main
from ue_astar import AStarEpisode
from utils.human_replay.converter import REPORT_PATH, convert, validate_output
from utils.human_replay.source import CAMERAS, HumanReplayEpisode, read_selection


def write_video(path: Path, count: int = 4, fps: int = 10):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for i in range(count):
            frame = av.VideoFrame.from_ndarray(
                np.full((24, 32, 3), 30 + 30 * i, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def dump(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_input(tmp_path: Path, count: int = 1):
    root = tmp_path / "input"
    root.mkdir()
    annotation = {
        "job_fingerprint": "job-fingerprint",
        "model": "qwen3.7-plus",
        "prompt_version": "nav_full_trajectory_v2",
        "sampling_fps": 10,
    }
    records, items = [], []
    for index in range(count):
        uid = f"00000000-0000-0000-0000-{index:012d}"
        directory = root / "episodes" / uid
        directory.mkdir(parents=True)
        poses = [
            [100, 200, 50, 0, 0, 30],
            [100, 200, 50, 0, 0, 30],
            [120, 200, 50, 0, 0, 60],
            [120, 180, 70, 0, 0, 60],
        ]
        k = [16, 0, 16, 0, 16, 12, 0, 0, 1]
        meta = {
            "episode_id": uid,
            "episode_index": 0,
            "status": "completed",
            "scene_id": "real-scene",
            "username": "collector",
            "sample_rate_hz": 10,
            "frame_count": 4,
            "capture_width": 32,
            "capture_height": 24,
            "camera_names": list(CAMERAS),
            **{f"K_{cam}": k for cam in CAMERAS},
            "rgb_video_paths": {cam: "Z:/stale/source.mp4" for cam in CAMERAS},
        }
        frames = []
        for i, pose in enumerate(poses):
            frame = {
                "frame_index": i,
                "pose": pose,
                "timestamp_us": 1000000 + i * i * 157000,
            }
            for cam, yaw in zip(CAMERAS, (0, 180, -90, 90)):
                frame[f"camera_pose_{cam}"] = [
                    pose[0],
                    pose[1],
                    pose[2] + 100,
                    0,
                    0,
                    pose[5] + yaw,
                ]
            frames.append(frame)
        dump(directory / "episode_meta.json", meta)
        dump(
            directory / "task_meta.json",
            {"instruction": "原任务不能作为训练指令", "task_type": "point_nav"},
        )
        replay = {
            "source": {"episodeId": uid, "revision": "revision-1"},
            "pipeline": {"fingerprint": "pipeline-1"},
            "media": {"fps": 10, "frameCount": 4},
            "trajectory": {"frameCount": 4, "durationSec": 7.5},
        }
        dump(directory / "replay.json", replay)
        (directory / "frames.jsonl").write_text(
            "".join(json.dumps(frame) + "\n" for frame in frames)
        )
        for cam in CAMERAS:
            write_video(directory / f"rgb/{cam}.mp4")
        record = {
            "episode_id": uid,
            "episode_relative_path": f"episodes/{uid}",
            "status": "prepared",
            "source_revision": "revision-1",
            "pipeline_fingerprint": "pipeline-1",
            "frame_count": 4,
            "fps": 10,
            "scene_id": "real-scene",
        }
        instruction = {
            "schema": "navigation_instruction",
            "version": 2,
            "source": {
                key: record[key]
                for key in (
                    "episode_id",
                    "episode_relative_path",
                    "source_revision",
                    "pipeline_fingerprint",
                )
            },
            "annotation": {
                "run_id": "selected-run",
                **{k: v for k, v in annotation.items() if k != "sampling_fps"},
                "annotation_sampling_fps": 10,
                "source_frame_count": 4,
                "camera": "front",
                "trajectory_usage": "full",
            },
            "has_quality_issue": False,
            "quality_reason": None,
            "vln": {"instruction": f"Walk forward and stop beside door {index}."},
            "objectnav": None,
        }
        dump(directory / "instruction.json", instruction)
        identity = hashlib.sha256(b"revision-1pipeline-1").hexdigest()[:16]
        items.append(
            {
                "item_id": f"human-{uid}-{identity}",
                "status": "instruction_written",
                "output_path": str(directory / "instruction.json"),
                "error": None,
            }
        )
        records.append(record)
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    report = tmp_path / "annotation-report.json"
    dump(
        report,
        {
            "schema": "navigation_process_report",
            "version": 1,
            "source": "human-replay",
            "task": "vln-generate",
            "run_id": "selected-run",
            "annotation": annotation,
            "total": count,
            "items": items,
        },
    )
    return root, report, records


def test_missing_annotation_blocks_full_export_but_explicit_subset_is_bounded(tmp_path):
    root, report, records = fixture_input(tmp_path, count=2)
    payload = json.loads(report.read_text())
    payload["items"][1].update(status="api_failed", output_path=None)
    dump(report, payload)
    selection = read_selection(root, report)
    assert len(selection.accepted) == len(selection.failed) == 1
    assert selection.excluded == []
    # Other episodes' media/frames must never be opened for an explicit subset.
    (root / records[1]["episode_relative_path"] / "frames.jsonl").write_text("invalid")
    subset = read_selection(root, report, [records[0]["episode_id"]])
    assert len(subset.accepted) == 1 and subset.failed == []
    output = tmp_path / "not-published"
    assert (
        main(
            [
                "--input-root",
                str(root),
                "--annotation-report",
                str(report),
                "--output-root",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()
    failure = json.loads(
        (output.with_name(output.name + ".staging") / REPORT_PATH).read_text()
    )
    assert failure["num_failed"] == 1 and failure["num_excluded"] == 0


def test_quality_filtered_is_distinct_and_requires_matching_report(tmp_path):
    root, report, records = fixture_input(tmp_path)
    path = root / records[0]["episode_relative_path"] / "instruction.json"
    payload = json.loads(path.read_text())
    payload.update(has_quality_issue=True, quality_reason="Endpoint obscured", vln=None)
    dump(path, payload)
    assert len(read_selection(root, report).failed) == 1
    report_data = json.loads(report.read_text())
    report_data["items"][0]["status"] = "filtered"
    dump(report, report_data)
    selected = read_selection(root, report)
    assert selected.accepted == selected.failed == []
    assert selected.excluded[0]["reason"] == "annotation_quality_issue"


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("annotation", "run_id", "other-run"),
        ("annotation", "job_fingerprint", "other-job"),
        ("source", "source_revision", "other-revision"),
        ("annotation", "source_frame_count", 3),
    ],
)
def test_wrong_annotation_provenance_is_rejected(tmp_path, section, key, value):
    root, report, records = fixture_input(tmp_path)
    path = root / records[0]["episode_relative_path"] / "instruction.json"
    payload = json.loads(path.read_text())
    payload[section][key] = value
    dump(path, payload)
    assert len(read_selection(root, report).failed) == 1


def test_human_and_astar_emit_identical_pose_and_preserve_duplicates(tmp_path):
    root, report, _ = fixture_input(tmp_path)
    record = read_selection(root, report).accepted[0]
    human = HumanReplayEpisode(root, record, 0)
    astar = AStarEpisode(
        episode_dir=human.episode_dir,
        meta=human.meta,
        frames=human.frames,
        camera_keys=list(CAMERAS),
        task=human.task,
        task_idx=0,
        task_info=[],
        body_from_camera=human.body_from_camera,
        astar_context={"graph": {"width": 3, "height": 3}},
        instruction_type="vln",
    )
    human_frames = [frame for frame, task in human]
    astar_frames = [frame for frame, task in astar]
    assert len(human_frames) == 4
    for first, second in zip(human_frames, astar_frames):
        np.testing.assert_array_equal(
            first["observation.state"], second["observation.state"]
        )
        np.testing.assert_array_equal(first["action"], second["action"])
    np.testing.assert_array_equal(human_frames[0]["action"], human_frames[1]["action"])
    assert human.metadata["scene_id"] == "real-scene"
    assert human.metadata["user_id"] == "collector"
    assert human.metadata["playback_duration_seconds"] == 0.4
    assert human.metadata["source_duration_seconds"] == 7.5


@pytest.mark.parametrize(
    "fault",
    ["index", "count", "pose", "intrinsic", "extrinsic", "video_count", "video_fps"],
)
def test_source_structure_errors_are_not_silently_trimmed_or_filtered(tmp_path, fault):
    root, report, records = fixture_input(tmp_path)
    directory = root / records[0]["episode_relative_path"]
    path = directory / "frames.jsonl"
    frames = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == "index":
        frames[1]["frame_index"] = 4
    if fault == "count":
        frames.append(frames[-1])
    if fault == "pose":
        frames[1]["pose"][0] = float("nan")
    if fault == "intrinsic":
        frames[1]["K_front"] = [17, 0, 16, 0, 16, 12, 0, 0, 1]
    if fault == "extrinsic":
        frames[1]["camera_pose_front"][0] += 20
    path.write_text("".join(json.dumps(frame) + "\n" for frame in frames))
    if fault == "video_count":
        write_video(directory / "rgb/rear.mp4", count=3)
    if fault == "video_fps":
        write_video(directory / "rgb/rear.mp4", fps=30)
    record = read_selection(root, report).accepted[0]
    with pytest.raises(ValueError):
        HumanReplayEpisode(root, record, 0)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_real_writer_publishes_complete_four_view_dataset_and_rejects_overwrite(
    tmp_path,
    num_workers,
):
    root, report, records = fixture_input(tmp_path, count=2)
    output = tmp_path / "lerobot-output"
    summary = convert(root, report, output, num_workers=num_workers)
    assert summary["status"] == "completed"
    assert summary["num_successful"] == 2 and summary["total_frames"] == 8
    assert not output.with_name(output.name + ".staging").exists()
    info = json.loads((output / "meta/info.json").read_text())
    assert info["fps"] == 10 and info["total_videos"] == 8
    modality = json.loads((output / "meta/modality.json").read_text())
    assert set(modality) == {"state", "action", "video", "annotation"}
    for index in range(2):
        table = pq.read_table(output / f"data/chunk-000/episode_{index:06d}.parquet")
        np.testing.assert_allclose(table["timestamp"].to_numpy(), [0, 0.1, 0.2, 0.3])
        assert table["index"].to_pylist() == list(range(index * 4, index * 4 + 4))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(output.name, root=output, video_backend="pyav")
    assert len(dataset) == 8
    sample = dataset[4]
    assert sample["video.front"].shape == (3, 24, 32)
    source_id = next(
        row["source_episode_id"]
        for row in summary["successful_episodes"]
        if row["episode_index"] == 1
    )
    assert (
        sample["task"] == f"Walk forward and stop beside door {int(source_id[-12:])}."
    )
    with pytest.raises(FileExistsError):
        convert(root, report, output)
    # Missing encoded outputs must fail even when metadata says the episode succeeded.
    (output / "videos/chunk-000/video.rear/episode_000001.mp4").rename(
        output / "saved-rear.mp4"
    )
    with pytest.raises((OSError, ValueError)):
        validate_output(output, read_selection(root, report).accepted, (24, 32))


def fail_encoding(**_):
    raise ValueError("simulated encoder failure")


def test_encoder_failure_does_not_publish_a_dataset_with_complete_metadata(
    tmp_path, monkeypatch
):
    from utils.lerobot import lerobot_creater

    root, report, _ = fixture_input(tmp_path)
    output = tmp_path / "not-published"
    monkeypatch.setattr(lerobot_creater, "encode_video_frames", fail_encoding)
    with pytest.raises((OSError, ValueError)):
        convert(root, report, output)
    assert not output.exists()
    staging = output.with_name(output.name + ".staging")
    assert (staging / "meta/episodes_extras.jsonl").exists()
    assert json.loads((staging / REPORT_PATH).read_text())["status"] == "failed"
