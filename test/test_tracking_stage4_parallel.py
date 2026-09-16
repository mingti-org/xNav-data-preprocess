from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import pytest

import tracking
from test.test_tracking import write_stage4_processed


SOURCE = "seed_9301/synthetic_scene/episode_000"


def convert(processed, output, *, workers=2, overwrite=False):
    return tracking.convert_four_view_dataset(
        processed, output, output.parent / f"{output.name}-work",
        workers=workers, overwrite=overwrite,
    )


def write_pair(root):
    write_stage4_processed(
        root, source_episode="seed_9301/synthetic_scene/episode_010",
        instruction="Follow B", commands=[[0.1, 0.0, 0.0]] * 2,
    )
    return write_stage4_processed(
        root, source_episode="seed_9301/synthetic_scene/episode_002",
        instruction="Follow A",
        commands=[[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.1, 0.0]],
    )


def test_metadata_index_does_not_scan_images_or_jsonl_tail(tmp_path, monkeypatch):
    processed = write_stage4_processed(tmp_path)
    jsonl = processed / f"jsonl/{SOURCE}.jsonl"
    lines = jsonl.read_text().splitlines()
    jsonl.write_text(lines[0] + "\ninvalid JSON in later rows\n")

    def unexpected_image_read(*args, **kwargs):
        pytest.fail("metadata indexing must not open an image")

    monkeypatch.setattr(tracking.Image, "open", unexpected_image_read)
    sources, _ = tracking.scan_stage4_inventory(processed)
    assert len(sources) == 1
    assert sources[0].plan.length == 3
    with pytest.raises(json.JSONDecodeError):
        convert(processed, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_workers_validate_concurrently_and_open_each_image_once(tmp_path, monkeypatch):
    processed = write_pair(tmp_path)
    real_load = tracking._load_stage4_episode
    real_open = tracking.Image.open
    real_resolve = tracking._resolve_stage4_relative
    barrier = threading.Barrier(2, timeout=15)
    image_reads = Counter()
    camera_resolves = Counter()
    lock = threading.Lock()

    def synchronized_load(source, root):
        assert threading.current_thread() is not threading.main_thread()
        barrier.wait()
        return real_load(source, root)

    def counted_open(path, *args, **kwargs):
        assert threading.current_thread() is not threading.main_thread()
        with lock:
            image_reads[Path(path)] += 1
        return real_open(path, *args, **kwargs)

    def counted_resolve(root, value, label):
        if label == "row camera metadata path":
            with lock:
                camera_resolves[value] += 1
        return real_resolve(root, value, label)

    monkeypatch.setattr(tracking, "_load_stage4_episode", synchronized_load)
    monkeypatch.setattr(tracking.Image, "open", counted_open)
    monkeypatch.setattr(tracking, "_resolve_stage4_relative", counted_resolve)
    result = convert(processed, tmp_path / "result")
    assert result["summary"] == {"train": {"episodes": 2, "tasks": 2, "frames": 6}}
    assert image_reads == Counter({path: 1 for path in processed.glob("frames/*/*/*/*/*.jpg")})
    assert len(image_reads) == 24
    assert len(camera_resolves) == 2 and set(camera_resolves.values()) == {1}


def test_out_of_order_completion_preserves_all_output_bytes(tmp_path, monkeypatch, caplog):
    processed = write_pair(tmp_path)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    convert(processed, serial, workers=1)
    real_process = tracking._process_four_view_episode
    later_finished = threading.Event()
    completion_order = []

    def reverse_completion(source, *args):
        if source.plan.episode_index == 0:
            assert later_finished.wait(timeout=15)
        result = real_process(source, *args)
        completion_order.append(source.plan.episode_index)
        if source.plan.episode_index == 1:
            later_finished.set()
        return result

    monkeypatch.setattr(tracking, "_process_four_view_episode", reverse_completion)
    with caplog.at_level(logging.INFO):
        convert(processed, parallel)
    assert completion_order == [1, 0]
    assert "Converted 2/2 episodes, 6/6 frames" in caplog.text
    assert "Validating final dataset metadata" in caplog.text
    original = {path.relative_to(serial): path.read_bytes() for path in serial.rglob("*") if path.is_file()}
    actual = {path.relative_to(parallel): path.read_bytes() for path in parallel.rglob("*") if path.is_file()}
    assert original == actual
    table = pq.read_table(parallel / "data/chunk-000/episode_000000.parquet")
    poses = np.array(table[tracking.ACTION_KEY].to_pylist())
    np.testing.assert_array_equal(poses[1], poses[2])
    np.testing.assert_array_equal(poses[2], poses[3])


@pytest.mark.parametrize("problem", [
    "missing_image", "corrupt_image", "wrong_size", "wrong_frame_index", "wrong_time",
    "wrong_episode", "changed_instruction", "bad_action", "missing_row", "wrong_current",
    "duplicate_image", "wrong_view", "camera_mismatch", "absolute_path", "parent_path",
    "symlink_escape", "symlink_wrong_view", "wrong_sample_count",
])
def test_invalid_episode_never_publishes_dataset(tmp_path, problem):
    processed = write_stage4_processed(tmp_path)
    jsonl = processed / f"jsonl/{SOURCE}.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    image_path = processed / rows[-1]["current_views"]["right"]
    if problem == "missing_image":
        image_path.unlink()
    elif problem == "corrupt_image":
        image_path.write_bytes(b"not a JPEG")
    elif problem == "wrong_size":
        Image.new("RGB", (32, 24)).save(image_path)
    elif problem == "wrong_frame_index":
        rows[-1]["frame_index"] = 10
    elif problem == "wrong_time":
        rows[-1]["sim_time_s"] = 9.0
    elif problem == "wrong_episode":
        rows[-1]["episode_id"] = "other_episode"
    elif problem == "changed_instruction":
        rows[-1]["instruction"] = "Follow another person"
    elif problem == "bad_action":
        rows[-1]["actions"] = [[float("nan"), 0, 0]]
    elif problem == "missing_row":
        rows.pop()
    elif problem == "wrong_current":
        rows[-1]["current"] = rows[-1]["current_views"]["left"]
    elif problem == "duplicate_image":
        rows[-1]["current_views"]["right"] = rows[0]["current_views"]["right"]
    elif problem == "wrong_view":
        rows[-1]["current_views"]["right"] = rows[-1]["current_views"]["left"]
    elif problem == "camera_mismatch":
        rows[-1]["camera_metadata"] = f"metadata/{SOURCE}/other_camera.json"
    elif problem == "absolute_path":
        rows[-1]["current_views"]["right"] = str(image_path)
    elif problem == "parent_path":
        rows[-1]["current_views"]["right"] = "../outside.jpg"
    elif problem == "symlink_escape":
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(image_path.read_bytes())
        image_path.unlink()
        image_path.symlink_to(outside)
    elif problem == "symlink_wrong_view":
        image_path.unlink()
        image_path.symlink_to(processed / rows[-1]["current_views"]["left"])
    elif problem == "wrong_sample_count":
        manifest = processed / f"metadata/{SOURCE}/source_manifest.json"
        data = json.loads(manifest.read_text())
        data["sample_count"] = 4
        manifest.write_text(json.dumps(data))
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows))

    output = tmp_path / "result"
    with pytest.raises((ValueError, OSError)):
        convert(processed, output)
    assert not output.exists()
    # All partial writes belong to this test's staging tree; none may be published.
    staging, = (tmp_path / "result-work").glob("result.stage4-staging-*")
    assert not list(staging.rglob("*.partial"))


def test_failed_conversion_preserves_existing_output(tmp_path):
    processed = write_stage4_processed(tmp_path)
    (processed / f"frames/{SOURCE}/back/frame_00003.jpg").unlink()
    output = tmp_path / "result"
    output.mkdir()
    marker = output / "existing_dataset"
    marker.write_bytes(b"preserve existing data")
    with pytest.raises(FileNotFoundError):
        convert(processed, output, overwrite=True)
    assert marker.read_bytes() == b"preserve existing data"
    assert list(output.iterdir()) == [marker]
