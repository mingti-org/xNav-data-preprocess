from __future__ import annotations

import json
import math

import numpy as np
import pyarrow.parquet as pq
import pytest

import tracking
from test.test_tracking import write_stage4_processed
from tracking_pose import measured_local_poses


def stt_row(position, heading):
    target = np.array([7.0, 2.0, -4.0])
    forward = np.array([math.cos(heading), -math.sin(heading)])
    direction = (target - position)[[0, 2]]
    error = math.atan2(forward[0] * direction[1] - forward[1] * direction[0], forward @ direction)
    return {"teacher": {"robot_position": list(position), "target_position": target.tolist(),
                        "yaw_error_rad": error}}


@pytest.mark.parametrize("mode", ["stt", "dt_at"])
def test_measured_coordinates_and_rotation(mode):
    # Initially facing -Z. Advance, slide left (+X), rise, then rotate in place.
    positions = [[2, 3, 4], [2, 3, 3.8], [2.1, 3.05, 3.8], [2.1, 3.05, 3.8]]
    headings = np.deg2rad([90, 90, 100, 110])
    if mode == "stt":
        rows = [stt_row(p, h) for p, h in zip(positions, headings)]
    else:
        rows = [{"teacher": {"dt_scene": {"robot_pose": {"position_m": p, "yaw_rad": h}}}}
                for p, h in zip(positions, headings)]
    poses, _ = measured_local_poses(rows)
    np.testing.assert_allclose(poses[:, :3], [[0, 0, 0], [.2, 0, 0], [.2, -.1, .05], [.2, -.1, .05]], atol=1e-7)
    np.testing.assert_allclose(2 * np.arctan2(poses[:, 5], poses[:, 6]), np.deg2rad([0, 0, 10, 20]), atol=1e-7)


def test_stt_crossing_pi_keeps_small_positive_rotation():
    poses, _ = measured_local_poses([stt_row([2, 3, 4], math.radians(h)) for h in [179, -179]])
    assert 2 * math.atan2(poses[1, 5], poses[1, 6]) == pytest.approx(math.radians(2))


@pytest.mark.parametrize("teacher", [{}, {"robot_position": [0, 0, 0], "target_position": [0, 0, 0], "yaw_error_rad": 0},
                                    {"dt_scene": {"robot_pose": {"position_m": [0, 0, 0], "yaw_rad": float('nan')}}}])
def test_missing_or_undefined_pose_fails_without_command_fallback(teacher):
    with pytest.raises(ValueError, match="measured pose at row 0"):
        measured_local_poses([{"teacher": teacher, "actions": [[1, 0, 0]]}])


def make_raw(root):
    source = "seed_9301/synthetic_scene/episode_000"
    processed = write_stage4_processed(root, commands=[[1, 0, .2]] * 3)
    episode = root / "raw" / source
    camera = json.loads((processed / "metadata" / source / "camera.json").read_text())
    camera["episode"] = {"scene": "synthetic_scene", "episode_id": "episode_000", "seed": 9301,
                         "instruction": "Follow the synthetic path"}
    for view, info in camera["views"].items():
        info["video_path"] = f"videos/{view}.mp4"
        tracking.encode_video_from_paths(
            episode / info["video_path"], sorted((processed / "frames" / source / view).glob("*.jpg")),
            width=info["width_px"], height=info["height_px"],
        )
    (episode / "camera.json").write_text(json.dumps(camera))
    (episode / "status.json").write_text(json.dumps({
        "state": "complete", "success": True, "frame_count": 3,
        "scene": "synthetic_scene", "episode_id": "episode_000", "seed": 9301,
    }))
    # Measured displacement is smaller than the requested forward action; then no movement.
    rows = [json.loads(line) for line in (episode / "steps.jsonl").read_text().splitlines()]
    for i, row in enumerate(rows):
        row.update(stt_row([2, 3, 4 - min(i, 1) * .1], math.pi / 2))
    (episode / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return episode


def test_raw_four_view_conversion_reuses_video_and_preserves_commands(tmp_path):
    episode = make_raw(tmp_path)
    output = tmp_path / "converted"
    result = tracking.convert_four_view_dataset(tmp_path / "raw", output, tmp_path / "work",
                                               raw_video_input=True, workers=2)
    assert result["summary"] == {"train": {"episodes": 1, "tasks": 1, "frames": 3}}
    assert (output / "meta/info.json").is_file() and not (output / "train").exists()
    table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    poses = np.array(table[tracking.STATE_KEY].to_pylist())
    np.testing.assert_allclose(poses[:, :3], [[0, 0, 0], [.1, 0, 0], [.1, 0, 0]], atol=1e-7)
    np.testing.assert_array_equal(table[tracking.ACTION_KEY].to_pylist(), poses)
    np.testing.assert_allclose(table[tracking.CONTROL_KEY].to_pylist(), [[1, 0, .2]] * 3)
    for key, view in tracking.STAGE4_VIDEO_VIEW_MAP.items():
        assert (output / f"videos/chunk-000/{key}/episode_000000.mp4").read_bytes() == (episode / f"videos/{view}.mp4").read_bytes()
    extra = json.loads((output / "meta/episodes_extras.jsonl").read_text())
    assert extra["pose_is_executed"] is True
    stats = json.loads((output / "meta/episodes_stats.jsonl").read_text())["stats"]
    assert stats[tracking.STATE_KEY]["max"][0] == pytest.approx(.1)


@pytest.mark.parametrize("fault", ["frame", "pose", "video"])
def test_raw_alignment_and_pose_fail_before_publication(tmp_path, fault):
    episode = make_raw(tmp_path)
    path = episode / "steps.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == "frame":
        rows[1]["video_frame_index"] = 2
    elif fault == "pose":
        del rows[1]["teacher"]["yaw_error_rad"]
    else:
        (episode / "videos/front.mp4").write_bytes(b"broken")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((ValueError, tracking.av.error.InvalidDataError)):
        tracking.convert_four_view_dataset(tmp_path / "raw", tmp_path / "output", tmp_path / "work", raw_video_input=True)
    assert not (tmp_path / "output").exists()


def test_stage4_raw_commands_must_match_frame(tmp_path):
    processed = write_stage4_processed(tmp_path)
    raw = next((tmp_path / "raw").glob("*/*/*/steps.jsonl"))
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    rows[1]["base_velocity_normalized"] = [0, 0, 0]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="raw/Stage-4 command mismatch"):
        tracking.convert_four_view_dataset(processed, tmp_path / "output", tmp_path / "work")
