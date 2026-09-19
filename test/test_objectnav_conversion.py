from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from ovon import EpisodeIterator, convert, discover, source_splits


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_video(path: Path, *, width: int = 64, height: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
    )
    assert writer.isOpened()
    try:
        for value in (32, 96, 160):
            writer.write(np.full((height, width, 3), value, dtype=np.uint8))
    finally:
        writer.release()


def _create_source(
    root: Path,
    *,
    dataset_name: str,
    layout: str,
) -> tuple[Path, Path]:
    if layout == "direct":
        split = root / "train"
    elif layout == "shard":
        split = root / "shard_0" / "train"
    elif layout == "hidden_shard":
        split = root / ".shards" / "0" / "train"
    else:
        raise ValueError(layout)
    episode_dir = split / "episodes" / "train" / "scene_traj_1"
    episode_dir.mkdir(parents=True)

    turn_left = Rotation.from_euler("y", 15.0, degrees=True).as_quat().tolist()
    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, -0.25], [0.0, 0.0, -0.25]]
    rotations = [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], turn_left]
    action_ids = [1, 2, 0]
    action_names = ["forward", "turn_left", "STOP"]
    steps = [
        {
            "step_index": index,
            "video_frame_index": index,
            "position": positions[index],
            "rotation": rotations[index],
            "discrete_action_to_next_id": action_ids[index],
            "discrete_action_to_next": action_names[index],
        }
        for index in range(3)
    ]
    instruction = "Find and go to the chair."
    episode = {
        "dataset": dataset_name,
        "role": None,
        "split": "train",
        "episode_id": "1",
        "trajectory_id": "1",
        "scene_key": "TestScene",
        "object_category": "chair",
        "instructions": [
            {
                "episode_id": "1",
                "trajectory_id": "1",
                "instruction": instruction,
            }
        ],
        "num_steps": 3,
        "num_frames": 3,
        "video_width": 64,
        "video_height": 48,
        "video_fps": 10,
        "video_hfov": 120.0,
        "metrics": {"success": True},
    }
    manifest = {
        **{
            key: episode[key]
            for key in (
                "dataset",
                "role",
                "split",
                "episode_id",
                "trajectory_id",
                "scene_key",
                "object_category",
            )
        },
        "episode_dir": "episodes/train/scene_traj_1",
        "episode_dir_name": "scene_traj_1",
        "num_steps": 3,
        "num_frames": 3,
        "success": True,
    }
    (episode_dir / "episode.json").write_text(json.dumps(episode), encoding="utf-8")
    _write_jsonl(episode_dir / "steps.jsonl", steps)
    np.savez(
        episode_dir / "trajectory.npz",
        positions=np.asarray(positions, dtype=np.float32),
        rotations=np.asarray(rotations, dtype=np.float32),
        discrete_action_to_next_ids=np.asarray(action_ids, dtype=np.int64),
        video_frame_indices=np.arange(3, dtype=np.int64),
    )
    for view in ("front", "back", "left", "right"):
        _write_video(episode_dir / f"{view}.mp4")
    _write_jsonl(split / "manifest.jsonl", [manifest])
    _write_jsonl(split / "errors.jsonl", [])
    return root, episode_dir


@pytest.mark.parametrize(
    ("layout", "partition"),
    [("direct", "direct"), ("shard", "shard_0"), ("hidden_shard", ".shards/0")],
)
def test_source_layout_is_discovered_from_directories(
    tmp_path: Path, layout: str, partition: str
) -> None:
    source_root, _ = _create_source(
        tmp_path / "source", dataset_name="hm3d_v2", layout=layout
    )
    sources, errors, splits = discover(source_root, "hm3d_v2")
    assert errors == []
    assert [split.partition for split in splits] == [partition]
    assert [source.source_key for source in sources] == [f"{partition}/scene_traj_1"]


def test_mixed_direct_and_sharded_layout_is_rejected(tmp_path: Path) -> None:
    source_root, _ = _create_source(
        tmp_path / "source", dataset_name="ovon", layout="direct"
    )
    (source_root / "shard_0" / "train").mkdir(parents=True)
    with pytest.raises(ValueError, match="both direct train and sharded"):
        source_splits(source_root)


def test_episode_iterator_converts_habitat_pose_to_xnav(tmp_path: Path) -> None:
    source_root, _ = _create_source(
        tmp_path / "source", dataset_name="ovon", layout="shard"
    )
    sources, errors, _ = discover(source_root, "ovon")
    assert errors == []
    frames = [frame for frame, _ in EpisodeIterator(sources[0], "ovon")]
    np.testing.assert_allclose(
        frames[0]["observation.state"], [0, 0, 0, 0, 0, 0, 1], atol=1e-6
    )
    np.testing.assert_allclose(
        frames[1]["observation.state"][:3], [0.25, 0, 0], atol=1e-6
    )
    yaw = Rotation.from_quat(frames[2]["observation.state"][3:]).as_euler(
        "ZYX", degrees=True
    )[0]
    assert yaw == pytest.approx(15.0)


def test_conversion_closes_parquet_video_and_provenance(tmp_path: Path) -> None:
    source_root, _ = _create_source(
        tmp_path / "source", dataset_name="ovon", layout="shard"
    )
    output_root = tmp_path / "processed"
    convert(source_root, output_root, "ovon", workers=2)

    report = json.loads((output_root / "conversion_report.json").read_text())
    assert report["publishable"] is True
    assert report["coordinate_frame"] == "xnav_episode_start_relative"
    assert report["source_layouts"] == ["shard_0"]
    extras = [
        json.loads(line)
        for line in (output_root / "meta/episodes_extras.jsonl").read_text().splitlines()
    ]
    assert extras[0]["source_episode_key"] == "shard_0/scene_traj_1"
    assert extras[0]["dataset_name"] == "ovon"
    table = pq.read_table(output_root / "data/chunk-000/episode_000000.parquet")
    states = np.asarray(table["observation.state"].to_pylist())
    np.testing.assert_allclose(states[1, :3], [0.25, 0, 0], atol=1e-6)
    assert table["action_text"].to_pylist() == ["forward", "turn_left", "STOP"]
    assert (output_root / "videos/chunk-000/video.rear/episode_000000.mp4").is_file()
    assert not (output_root / "videos/chunk-000/video.back").exists()


def test_worker_error_blocks_publication_and_is_reported(tmp_path: Path) -> None:
    source_root, episode_dir = _create_source(
        tmp_path / "source", dataset_name="ovon", layout="direct"
    )
    _write_video(episode_dir / "front.mp4", width=32, height=24)
    output_root = tmp_path / "processed"
    with pytest.raises(RuntimeError, match="child-process error"):
        convert(source_root, output_root, "ovon", workers=1)
    assert not output_root.exists()
    staging = Path(str(output_root) + ".staging")
    report = json.loads((staging / "conversion_report.json").read_text())
    assert report["publishable"] is False
    errors = [json.loads(line) for line in (staging / "errors.jsonl").read_text().splitlines()]
    assert errors[0]["stage"] == "episode_conversion"
    assert errors[0]["source_episode_key"] == "direct/scene_traj_1"
