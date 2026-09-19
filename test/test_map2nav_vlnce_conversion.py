from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from test.map2nav_vlnce_testdata import (
    create_replay_source,
    create_rxr_replay_source,
    read_jsonl,
    write_jsonl,
)
from utils.map2nav_vlnce import convert_dataset
from utils.map2nav_vlnce.filtering import SourceSchemaError
from utils.map2nav_vlnce.schema import MAP_ASSET_KEYS, PARQUET_COLUMNS, SCHEMA_VERSION


def test_conversion_filters_before_limit_and_writes_stable_copy(tmp_path: Path) -> None:
    source_root = create_replay_source(tmp_path / "source")
    output_root = tmp_path / "processed" / "r2r"

    dataset_root = convert_dataset(
        input_root=source_root,
        output_root=output_root,
        dataset_name="r2r",
        split="train",
        max_episodes=1,
        chunk_size=2,
    )

    assert dataset_root == output_root / "train"
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    report = json.loads(
        (dataset_root / "meta" / "conversion_report.json").read_text(encoding="utf-8")
    )
    extras = read_jsonl(dataset_root / "meta" / "episodes_extras.jsonl")
    skipped = read_jsonl(dataset_root / "meta" / "skipped_episodes.jsonl")
    tasks = read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    modality = json.loads(
        (dataset_root / "meta" / "modality.json").read_text(encoding="utf-8")
    )

    assert info["robot_type"] == "map2nav_vlnce"
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 2
    assert info["total_videos"] == 4
    assert info["splits"] == {"train": "0:1"}
    assert set(info["features"]) == set(PARQUET_COLUMNS) | {
        "video.front",
        "video.left",
        "video.right",
        "video.rear",
    }
    assert report["source_manifest_total"] == 3
    assert report["source_instruction_total"] == 9
    assert report["selected_instruction_total_before_floor_filter"] == 9
    assert report["eligible_single_floor"] == 2
    assert report["eligible_single_floor_with_selected_instructions"] == 2
    assert report["language_filtered_single_floor"] == 0
    assert report["eligible_instruction_episodes"] == 6
    assert report["accepted"] == 1
    assert report["skipped_multi_floor"] == 1
    assert report["skipped_multi_floor_selected_instructions"] == 3
    assert report["unconverted_eligible_instruction_episodes_due_to_limit"] == 5
    assert report["errors"] == 0
    assert skipped == [
        {
            "source_manifest_index": 0,
            "source_episode_dir": "episodes/train/mp3d_TestScene_traj_multi",
            "trajectory_id": "multi",
            "scene_key": "TestScene",
            "reason": "multi_floor",
            "visited_levels": [0, 1],
            "source_instruction_count": 3,
            "selected_instruction_count": 3,
        }
    ]
    assert tasks == [{"task_index": 0, "task": "instruction 0 for single_a"}]
    assert modality["state"]["drone"]["absolute"] is True
    assert modality["action"]["pose"]["absolute"] is True

    extra = extras[0]
    assert extra["schema_version"] == SCHEMA_VERSION
    assert extra["episode_index"] == 0
    assert extra["trajectory_id"] == "single_a"
    assert extra["role"] is None
    assert extra["instructions"] == [
        {
            "episode_id": "3",
            "trajectory_id": "single_a",
            "instruction": "instruction 0 for single_a",
        }
    ]
    assert set(extra["map_assets"]) == set(MAP_ASSET_KEYS)
    assert extra["map_projection"]["bounds_xz"] == [0.0, -9.0, 19.0, 0.0]
    np.testing.assert_allclose(
        extra["map_projection"]["world_xz_to_pixel"],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 9.0], [0.0, 0.0, 1.0]],
    )

    parquet_path = dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    table = pq.read_table(parquet_path)
    assert table.column_names == PARQUET_COLUMNS
    frame = pd.read_parquet(parquet_path)
    assert frame["extra.cot"].tolist() == ["", ""]
    assert "extra.floor_level_id" not in frame
    assert "extra.target_index" not in frame
    np.testing.assert_array_equal(frame.iloc[0]["observation.state"], [0, 0, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(frame.iloc[1]["observation.state"], [1, 0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(
        np.stack(frame["observation.state"]), np.stack(frame["action"])
    )
    np.testing.assert_array_equal(
        np.stack(frame["extra.habitat_world_position"]), [[0, 0, 0], [0, 0, -1]]
    )
    np.testing.assert_array_equal(
        np.stack(frame["extra.habitat_world_rotation_xyzw"]),
        [[0, 0, 0, 1], [0, 0, 0, 1]],
    )
    assert [value.tolist() for value in frame["extra.discrete_action_to_next_id"]] == [[1], [0]]

    source_split = source_root / "train"
    for key, relative in extra["map_assets"].items():
        target = dataset_root / relative
        assert target.parent == dataset_root / "maps" / "chunk-000" / "episode_000000"
        source_relative = {
            "graph": "scenes/TestScene/graph_floor_0p000/graph.png",
            "graph_overlay": "episodes/train/mp3d_TestScene_traj_single_a/overlays/trajectory_on_graph_floor_0p000.png",
            "floorplan": "scenes/TestScene/level_0/layout.png",
            "floorplan_overlay": "episodes/train/mp3d_TestScene_traj_single_a/overlays/trajectory_on_layout_level_0.png",
            "floorplan_detail": "scenes/TestScene/level_0/detail.png",
            "floorplan_detail_overlay": "episodes/train/mp3d_TestScene_traj_single_a/overlays/trajectory_on_detail_level_0.png",
        }[key]
        source = source_split / source_relative
        assert target.read_bytes() == source.read_bytes()
        assert target.stat().st_ino != source.stat().st_ino

    for source_view, target_view in {
        "front": "front",
        "left": "left",
        "right": "right",
        "back": "rear",
    }.items():
        source = source_split / extra["source_episode_dir"] / f"{source_view}.mp4"
        target = (
            dataset_root
            / "videos"
            / "chunk-000"
            / f"video.{target_view}"
            / "episode_000000.mp4"
        )
        assert target.read_bytes() == source.read_bytes()
        assert target.stat().st_ino != source.stat().st_ino


def test_resume_reuses_completed_episode_files(tmp_path: Path) -> None:
    source_root = create_replay_source(tmp_path / "source")
    output_root = tmp_path / "processed" / "r2r"
    dataset_root = convert_dataset(
        source_root, output_root, "r2r", "train", max_episodes=1
    )
    video = dataset_root / "videos" / "chunk-000" / "video.front" / "episode_000000.mp4"
    before = (video.stat().st_ino, video.stat().st_mtime_ns)

    resumed_root = convert_dataset(
        source_root,
        output_root,
        "r2r",
        "train",
        max_episodes=1,
        resume=True,
    )

    assert resumed_root == dataset_root
    assert (video.stat().st_ino, video.stat().st_mtime_ns) == before


def test_parallel_conversion_preserves_accepted_manifest_order(tmp_path: Path) -> None:
    source_root = create_replay_source(tmp_path / "source")
    dataset_root = convert_dataset(
        source_root,
        tmp_path / "processed" / "r2r",
        "r2r",
        "train",
        num_workers=2,
    )

    extras = read_jsonl(dataset_root / "meta" / "episodes_extras.jsonl")
    tasks = read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    assert [row["episode_index"] for row in extras] == list(range(6))
    assert [row["trajectory_id"] for row in extras] == ["single_a"] * 3 + ["single_b"] * 3
    assert [row["source_episode_dir"] for row in extras] == [
        "episodes/train/mp3d_TestScene_traj_single_a",
    ] * 3 + [
        "episodes/train/mp3d_TestScene_traj_single_b",
    ] * 3
    assert tasks == [
        {"task_index": index, "task": f"instruction {index % 3} for " + ("single_a" if index < 3 else "single_b")}
        for index in range(6)
    ]


def test_resume_rejects_a_changed_chunk_size(tmp_path: Path) -> None:
    source_root = create_replay_source(tmp_path / "source")
    output_root = tmp_path / "processed" / "r2r"
    convert_dataset(source_root, output_root, "r2r", "train", chunk_size=2)

    with pytest.raises(SourceSchemaError, match="chunk_size"):
        convert_dataset(
            source_root,
            output_root,
            "r2r",
            "train",
            chunk_size=1,
            resume=True,
        )


def test_resume_rejects_shrinking_below_completed_episode_count(tmp_path: Path) -> None:
    source_root = create_replay_source(tmp_path / "source")
    output_root = tmp_path / "processed" / "r2r"
    convert_dataset(source_root, output_root, "r2r", "train")

    with pytest.raises(SourceSchemaError, match="max_episodes"):
        convert_dataset(
            source_root,
            output_root,
            "r2r",
            "train",
            max_episodes=1,
            resume=True,
        )


def test_preflight_schema_error_creates_no_output_split(tmp_path: Path) -> None:
    source_root = create_replay_source(tmp_path / "source")
    bad_steps = (
        source_root
        / "train"
        / "episodes"
        / "train"
        / "mp3d_TestScene_traj_multi"
        / "steps.jsonl"
    )
    rows = [
        json.loads(line)
        for line in bad_steps.read_text(encoding="utf-8").splitlines()
        if line
    ]
    del rows[0]["floor_level_id"]
    bad_steps.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    output_root = tmp_path / "processed" / "r2r"

    with pytest.raises(SourceSchemaError, match="floor_level_id"):
        convert_dataset(source_root, output_root, "r2r", "train")

    assert not (output_root / "train").exists()


def test_conversion_ignores_recorded_source_errors_and_reports_count(
    tmp_path: Path,
) -> None:
    source_root = create_replay_source(tmp_path / "source")
    write_jsonl(
        source_root / "train" / "errors.jsonl",
        [{"episode_dir_name": "failed_trajectory", "error": "GreedyFollowerError()"}],
    )

    dataset_root = convert_dataset(
        source_root,
        tmp_path / "processed" / "r2r",
        "r2r",
        "train",
    )

    report = json.loads(
        (dataset_root / "meta" / "conversion_report.json").read_text(encoding="utf-8")
    )
    assert report["source_manifest_total"] == 3
    assert report["source_recorded_errors"] == 1
    assert report["complete_success_manifest_conversion"] is True
    assert report["complete_source_conversion"] is False


def test_rxr_conversion_joins_language_metadata_and_keeps_only_english(
    tmp_path: Path,
) -> None:
    source_root, annotation_path = create_rxr_replay_source(tmp_path / "source")
    output_root = tmp_path / "processed" / "rxr_guide"

    dataset_root = convert_dataset(
        source_root,
        output_root,
        "rxr_guide",
        "train",
        rxr_annotations=annotation_path,
        num_workers=2,
    )

    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    report = json.loads(
        (dataset_root / "meta" / "conversion_report.json").read_text(encoding="utf-8")
    )
    tasks = read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    extras = read_jsonl(dataset_root / "meta" / "episodes_extras.jsonl")
    skipped = read_jsonl(dataset_root / "meta" / "skipped_episodes.jsonl")

    assert info["total_episodes"] == 2
    assert info["total_tasks"] == 2
    assert info["total_frames"] == 4
    assert report["source_manifest_total"] == 3
    assert report["source_instruction_total"] == 10
    assert report["selected_instruction_total_before_floor_filter"] == 4
    assert report["eligible_single_floor"] == 2
    assert report["eligible_single_floor_with_selected_instructions"] == 1
    assert report["language_filtered_single_floor"] == 1
    assert report["eligible_instruction_episodes"] == 2
    assert report["eligible_instruction_language_counts"] == {"en-IN": 1, "en-US": 1}
    assert report["accepted_instruction_language_counts"] == {"en-IN": 1, "en-US": 1}
    assert report["skipped_multi_floor"] == 1
    assert report["skipped_multi_floor_selected_instructions"] == 2
    assert tasks == [
        {"task_index": 0, "task": "en-US instruction for single_a"},
        {"task_index": 1, "task": "en-IN instruction for single_a"},
    ]
    assert [row["instructions"][0]["language"] for row in extras] == ["en-US", "en-IN"]
    assert [row["instructions"][0]["episode_id"] for row in extras] == ["1004", "1005"]
    assert [row["reason"] for row in skipped] == ["multi_floor", "no_selected_instruction"]
    assert skipped[1]["source_languages"] == ["hi-IN", "te-IN"]


def test_rxr_conversion_accepts_rxr_en_guide_source_identity(tmp_path: Path) -> None:
    source_root, annotation_path = create_rxr_replay_source(tmp_path / "source")
    split_root = source_root / "train"
    manifest = read_jsonl(split_root / "manifest.jsonl")
    for row in manifest:
        row["dataset"] = "rxr_en"
        episode_path = split_root / row["episode_dir"] / "episode.json"
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        episode["dataset"] = "rxr_en"
        episode_path.write_text(json.dumps(episode), encoding="utf-8")
    write_jsonl(split_root / "manifest.jsonl", manifest)

    dataset_root = convert_dataset(
        source_root,
        tmp_path / "processed" / "rxr_guide",
        "rxr_guide",
        "train",
        max_episodes=1,
        rxr_annotations=annotation_path,
    )

    extras = read_jsonl(dataset_root / "meta" / "episodes_extras.jsonl")
    assert extras[0]["dataset_name"] == "rxr_guide"
    assert extras[0]["role"] == "guide"


def test_rxr_conversion_requires_authoritative_language_annotations(tmp_path: Path) -> None:
    source_root, _ = create_rxr_replay_source(tmp_path / "source")
    output_root = tmp_path / "processed" / "rxr_guide"

    with pytest.raises(SourceSchemaError, match="authoritative RxR guide annotation"):
        convert_dataset(source_root, output_root, "rxr_guide", "train")

    assert not (output_root / "train").exists()


@pytest.mark.parametrize("dataset_name", ["r2r", "rxr_guide"])
def test_existing_datasets_still_require_floorplan_assets(
    tmp_path: Path, dataset_name: str
) -> None:
    kwargs = {}
    if dataset_name == "rxr_guide":
        source_root, annotations = create_rxr_replay_source(tmp_path / "source")
        kwargs["rxr_annotations"] = annotations
    else:
        source_root = create_replay_source(tmp_path / "source")
    (source_root / "train/scenes/TestScene/level_0/detail.png").unlink()
    with pytest.raises(SourceSchemaError, match="floorplan_detail map asset"):
        convert_dataset(
            source_root, tmp_path / "processed", dataset_name, "train", **kwargs,
        )


def test_rxr_cli_accepts_authoritative_annotations(tmp_path: Path) -> None:
    source_root, annotation_path = create_rxr_replay_source(tmp_path / "source")
    output_root = tmp_path / "processed" / "rxr_guide"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "map2nav_vlnce.py"),
        "--input-root",
        str(source_root),
        "--output-root",
        str(output_root),
        "--dataset-name",
        "rxr_guide",
        "--split",
        "train",
        "--rxr-annotations",
        str(annotation_path),
        "--max-episodes",
        "1",
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    tasks = read_jsonl(output_root / "train" / "meta" / "tasks.jsonl")
    assert tasks == [{"task_index": 0, "task": "en-US instruction for single_a"}]


def test_map2nav_vlnce_cli_uses_flat_copy_only_contract(tmp_path: Path) -> None:
    source_root = create_replay_source(tmp_path / "source")
    output_root = tmp_path / "processed" / "r2r"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "map2nav_vlnce.py"),
        "--input-root",
        str(source_root),
        "--output-root",
        str(output_root),
        "--dataset-name",
        "r2r",
        "--split",
        "train",
        "--flat-output",
        "--max-episodes",
        "1",
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "map2nav_vlnce" in result.stdout
    assert (output_root / "meta" / "conversion_report.json").is_file()
    assert not (output_root / "train").exists()
    context = json.loads(
        (output_root / "meta" / ".conversion" / "context.json").read_text(
            encoding="utf-8"
        )
    )
    assert context["output_layout"] == "flat"


def test_map2nav_wrapper_runs_both_flat_train_exports_with_32_workers() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "map2nav_vlnce.sh"
    ).read_text(encoding="utf-8")

    assert "MAP2NAV_NUM_WORKERS=32" in script
    assert script.count("--flat-output") == 2
    assert script.count("--split train") == 2
    assert "processed_v2/r2r" not in script
    assert '${processed_root}/r2r' in script
    assert '${processed_root}/rxr' in script
    assert script.index("r2r_replay_4_view_2048") < script.index(
        "rxr_replay_guide_4_view_2048"
    )
