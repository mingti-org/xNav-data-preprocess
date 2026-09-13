from __future__ import annotations

import importlib.util
import io
import json
import sqlite3
import tarfile
from fractions import Fraction
from pathlib import Path

import av
import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_human_replay", Path(__file__).parents[1] / "scripts/prepare_human_replay.py"
)
preparer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparer)
UID = "00000000-0000-0000-0000-000000000001"


def make_source(tmp_path, *, bad_index=False, unsafe=False):
    root = tmp_path / "source"
    root.mkdir()
    members = tmp_path / "members"
    members.mkdir()
    meta = {
        "episode_id": UID,
        "status": "completed",
        "sample_rate_hz": 10,
        "frame_count": 4,
        "scene_id": "scene",
    }
    replay = {
        "source": {"episodeId": UID, "revision": "revision-1"},
        "pipeline": {"fingerprint": "pipeline-1"},
        "media": {"frameCount": 4},
        "trajectory": {"frameCount": 4},
    }
    for name, value in (
        ("episode_meta.json", meta),
        ("replay.json", replay),
        ("task_meta.json", {}),
        ("rgb/meta.json", {}),
    ):
        p = members / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
    (members / "events.jsonl").write_text("")
    frames = []
    for index in range(4):
        frames.append(
            {
                "frame_index": index + (1 if bad_index else 0),
                "pose": [index, 0, 0, 0, 0, 0],
                **{f"camera_pose_{cam}": [index, 0, 0, 0, 0, 0] for cam in preparer.CAMERAS},
            }
        )
    (members / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    for cam in preparer.CAMERAS:
        with av.open(str(members / f"rgb/{cam}.mp4"), "w") as container:
            stream = container.add_stream("libx264", rate=10)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for index in range(4):
                frame = av.VideoFrame(64, 48, "rgb24")
                frame.planes[0].update(bytes([index * 20, 40, 80]) * 64 * 48)
                frame.pts, frame.time_base = index, Fraction(1, 10)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    archive = root / f"{UID}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(members.rglob("*")):
            if path.is_file():
                handle.add(path, arcname=path.relative_to(members).as_posix())
        extra = tarfile.TarInfo("../escaped" if unsafe else "depth/front.mp4")
        extra.size = 5
        handle.addfile(extra, io.BytesIO(b"depth"))
    row = {
        "episode_id": UID,
        "source_revision": "revision-1",
        "archive_size_bytes": archive.stat().st_size,
    }
    return root, archive, row


def test_preparation_preserves_short_frames_and_resumes_without_overwrite(tmp_path):
    root, archive, row = make_source(tmp_path)
    output = tmp_path / "prepared"
    original = archive.read_bytes()
    first = preparer.prepare_episode(root, output, row)
    assert first["frame_count"] == 4
    episode = output / "episodes" / UID
    (episode / "instruction.json").write_text("existing annotation")
    assert preparer.prepare_episode(root, output, row) == first
    assert (episode / "instruction.json").read_text() == "existing annotation"
    assert not (episode / "depth").exists()
    assert archive.read_bytes() == original


@pytest.mark.parametrize(
    "kwargs,pattern", [({"bad_index": True}, "Non-contiguous"), ({"unsafe": True}, "Unsafe")]
)
def test_invalid_archive_never_publishes_episode(tmp_path, kwargs, pattern):
    root, _, row = make_source(tmp_path, **kwargs)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match=pattern):
        preparer.prepare_episode(root, output, row)
    assert not (output / "episodes" / UID).exists()
    assert not (output / "episodes/escaped").exists()


def test_manifest_selection_uses_replay_metadata_and_ignores_windows_path(tmp_path):
    root, archive, row = make_source(tmp_path)
    manifest = tmp_path / "manifest.sqlite"
    with sqlite3.connect(manifest) as conn:
        conn.execute(
            "CREATE TABLE episodes(episode_id TEXT, source_revision TEXT, "
            "archive_size_bytes INTEGER, metadata_json TEXT, state TEXT, archive_path TEXT)"
        )
        conn.execute(
            "INSERT INTO episodes VALUES(?,?,?,?,?,?)",
            (
                UID,
                row["source_revision"],
                archive.stat().st_size,
                json.dumps({"validity": "valid", "frameCount": 120, "sampleRateHz": 30}),
                "downloaded",
                "C:\\Replay\\unused.tar.gz",
            ),
        )
    output = tmp_path / "prepared"
    report = preparer.prepare(root, manifest, output)
    assert report["total_frames"] == 4
    assert json.loads((output / "manifest.jsonl").read_text())["fps"] == 10
