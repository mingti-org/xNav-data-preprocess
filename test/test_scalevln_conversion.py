from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from test.map2nav_vlnce_testdata import (
    create_scalevln_replay_source,
    read_jsonl,
    write_jsonl,
)
from utils.map2nav_vlnce import convert_dataset
from utils.map2nav_vlnce.filtering import SourceSchemaError
from utils.map2nav_vlnce.schema import (
    PARQUET_COLUMNS,
    SCALEVLN_MAP_ASSET_KEYS,
    SCHEMA_VERSION,
)


@pytest.mark.parametrize("skip_preflight", [False, True])
def test_scalevln_conversion_uses_graph_only_map_contract(
    tmp_path: Path, skip_preflight: bool
) -> None:
    source_root = create_scalevln_replay_source(tmp_path / "source")
    output_root = tmp_path / "processed" / "scalevln"

    dataset_root = convert_dataset(
        input_root=source_root,
        output_root=output_root,
        dataset_name="scalevln",
        split="train",
        chunk_size=2,
        skip_preflight=skip_preflight,
    )

    assert dataset_root == output_root / "train"
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    report = json.loads(
        (dataset_root / "meta" / "conversion_report.json").read_text(encoding="utf-8")
    )
    extras = read_jsonl(dataset_root / "meta" / "episodes_extras.jsonl")
    tasks = read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    skipped = read_jsonl(dataset_root / "meta" / "skipped_episodes.jsonl")

    assert info["schema_version"] == SCHEMA_VERSION
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 4
    assert info["total_tasks"] == 2
    assert info["total_videos"] == 8
    assert info["splits"] == {"train": "0:2"}

    assert report["dataset_name"] == "scalevln"
    assert report["source_manifest_total"] == 2
    assert report["source_instruction_total"] == 2
    assert report["selected_instruction_total_before_floor_filter"] == 2
    assert report["eligible_source_episodes"] == 2
    assert report["floor_filter"] == "not_applied"
    assert report["floor_filter_reason"] == "source_has_no_floor_metadata"
    assert report["eligible_instruction_episodes"] == 2
    assert report["accepted"] == 2
    assert report["accepted_frames"] == 4
    assert report["accepted_instruction_language_counts"] == {}
    assert "eligible_single_floor" not in report
    assert "eligible_single_floor_with_selected_instructions" not in report
    assert "skipped_multi_floor" not in report
    assert "skipped_multi_floor_selected_instructions" not in report
    assert "language_filtered_single_floor" not in report
    assert skipped == []

    assert [task["task"] for task in tasks] == [
        "instruction for scalevln_a",
        "instruction for scalevln_b",
    ]

    first = extras[0]
    assert first["dataset_name"] == "scalevln"
    assert first["role"] is None
    assert first["instructions"] == [
        {
            "episode_id": "sc_0",
            "trajectory_id": "scalevln_a",
            "instruction": "instruction for scalevln_a",
        }
    ]
    assert first["video"]["hfov"] == 120.0

    # Only the graph assets are required and copied; the projection is derived
    # from the graph-floor metadata because ScaleVLN exports no floorplans.
    assert set(first["map_assets"]) == set(SCALEVLN_MAP_ASSET_KEYS)
    assert first["map_projection"]["bounds_xz"] == [0.0, -9.0, 19.0, 0.0]
    np.testing.assert_allclose(
        first["map_projection"]["world_xz_to_pixel"],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 9.0], [0.0, 0.0, 1.0]],
    )

    map_directory = dataset_root / "maps" / "chunk-000" / "episode_000000"
    assert sorted(path.name for path in map_directory.iterdir()) == [
        "graph.png",
        "graph_overlay.png",
    ]

    parquet_path = dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    assert pq.read_table(parquet_path).column_names == PARQUET_COLUMNS

    for view in ("front", "left", "right", "rear"):
        video = (
            dataset_root
            / "videos"
            / "chunk-000"
            / f"video.{view}"
            / "episode_000000.mp4"
        )
        assert video.is_file() and video.stat().st_size > 0


def test_scalevln_conversion_rejects_a_missing_graph_overlay(tmp_path: Path) -> None:
    source_root = create_scalevln_replay_source(tmp_path / "source")
    split_root = source_root / "train"
    manifest = read_jsonl(split_root / "manifest.jsonl")
    overlay = (
        split_root
        / manifest[0]["episode_dir"]
        / "overlays"
        / "trajectory_on_graph_floor_0p000.png"
    )
    overlay.unlink()

    with pytest.raises(SourceSchemaError, match="overlay"):
        convert_dataset(
            input_root=source_root,
            output_root=tmp_path / "processed" / "scalevln",
            dataset_name="scalevln",
            split="train",
        )


def test_scalevln_source_identity_is_enforced(tmp_path: Path) -> None:
    source_root = create_scalevln_replay_source(tmp_path / "source")

    with pytest.raises(SourceSchemaError, match="dataset/role mismatch"):
        convert_dataset(
            input_root=source_root,
            output_root=tmp_path / "processed" / "r2r",
            dataset_name="r2r",
            split="train",
        )


@pytest.mark.parametrize("skip_preflight", [False, True])
def test_scalevln_shards_share_output_indices_and_resume(
    tmp_path: Path, skip_preflight: bool
) -> None:
    source_root = tmp_path / "source"
    for shard_index in (1, 0):
        shard = f"shard_{shard_index}"
        create_scalevln_replay_source(source_root / shard, instruction_prefix=shard)
        split_root = source_root / shard / "train"
        # The same scene and episode directory names in different shards must
        # resolve to their own assets, even when workers complete out of order.
        Image.new("RGB", (20, 10), (shard_index * 100, 0, 0)).save(
            split_root / "scenes/TestScene/graph_floor_0p000/graph.png"
        )
        write_jsonl(split_root / "errors.jsonl", [{"episode_dir": "not_exported"}])

    output_root = tmp_path / "processed"
    dataset_root = convert_dataset(
        source_root, output_root, "scalevln", "train",
        chunk_size=2, num_workers=2, skip_preflight=skip_preflight,
    )
    extras = read_jsonl(dataset_root / "meta/episodes_extras.jsonl")
    assert [row["instructions"][0]["episode_id"] for row in extras] == [
        "shard_0_0", "shard_0_1", "shard_1_0", "shard_1_1",
    ]
    for index, extra in enumerate(extras):
        shard = f"shard_{index // 2}"
        assert extra["source_episode_dir"].startswith(f"{shard}/train/episodes/train/")
        graph = dataset_root / extra["map_assets"]["graph"]
        expected_graph = source_root / shard / "train/scenes/TestScene/graph_floor_0p000/graph.png"
        assert graph.read_bytes() == expected_graph.read_bytes()
        table = pq.read_table(
            dataset_root / f"data/chunk-{index // 2:03d}/episode_{index:06d}.parquet"
        )
        assert table["episode_index"].to_pylist() == [index, index]
        assert table["task_index"].to_pylist() == [index, index]
        assert table["index"].to_pylist() == [index * 2, index * 2 + 1]
        fragment = json.loads(
            (dataset_root / f"meta/.conversion/episodes/episode_{index:06d}.json").read_text()
        )
        assert fragment["source_manifest_index"] == index

    report = json.loads((dataset_root / "meta/conversion_report.json").read_text())
    assert report["source_manifest_total"] == report["accepted"] == 4
    assert report["accepted_frames"] == 8
    assert report["source_recorded_errors"] == 2
    assert report["complete_success_manifest_conversion"] is True
    assert report["complete_source_conversion"] is False
    context = json.loads((dataset_root / "meta/.conversion/context.json").read_text())
    assert context["source_splits"] == ["shard_0/train", "shard_1/train"]

    parquet_files = sorted((dataset_root / "data").glob("*/*.parquet"))
    mtimes = [path.stat().st_mtime_ns for path in parquet_files]
    convert_dataset(
        source_root, output_root, "scalevln", "train",
        chunk_size=2, num_workers=2, resume=True, skip_preflight=not skip_preflight,
    )
    assert [path.stat().st_mtime_ns for path in parquet_files] == mtimes
    assert read_jsonl(dataset_root / "meta/episodes_extras.jsonl") == extras

    create_scalevln_replay_source(source_root / "shard_2", instruction_prefix="shard_2")
    with pytest.raises(SourceSchemaError, match="resume context mismatch for source_splits"):
        convert_dataset(
            source_root, output_root, "scalevln", "train", chunk_size=2, resume=True,
        )


def test_scalevln_shards_reject_duplicate_instruction_ids_before_writing(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    for shard in ("shard_0", "shard_1"):
        create_scalevln_replay_source(source_root / shard)
    output_root = tmp_path / "processed"
    with pytest.raises(SourceSchemaError, match="not globally unique"):
        convert_dataset(source_root, output_root, "scalevln", "train")
    assert not output_root.exists()


@pytest.mark.parametrize("skip_preflight", [False, True])
def test_scalevln_does_not_infer_floors_but_still_checks_projection(
    tmp_path: Path, skip_preflight: bool
) -> None:
    source_root = create_scalevln_replay_source(tmp_path / "source")
    split_root = source_root / "train"
    manifest = read_jsonl(split_root / "manifest.jsonl")
    for row in manifest:
        path = split_root / row["episode_dir"] / "steps.jsonl"
        steps = read_jsonl(path)
        for index, step in enumerate(steps):
            step.pop("floor_level_id")
            step["position"][1] = float(index * 3)
        write_jsonl(path, steps)
    dataset_root = convert_dataset(
        source_root, tmp_path / "valid", "scalevln", "train", skip_preflight=skip_preflight,
    )
    report = json.loads((dataset_root / "meta/conversion_report.json").read_text())
    assert report["accepted"] == 2
    assert report["floor_filter"] == "not_applied"

    path = split_root / manifest[0]["episode_dir"] / "steps.jsonl"
    steps = read_jsonl(path)
    for key in ("map_xy", "graph_xy", "floorplan_xy"):
        steps[0][key] = [10, 9]
    write_jsonl(path, steps)
    with pytest.raises(SourceSchemaError, match="projection differs"):
        convert_dataset(
            source_root, tmp_path / "invalid", "scalevln", "train",
            skip_preflight=skip_preflight,
        )
