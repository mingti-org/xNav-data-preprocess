#!/usr/bin/env python3
"""Prepare UUID replay archives without changing frames, poses, or the source."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import tarfile
import uuid
from pathlib import Path, PurePosixPath

CAMERAS = ("front", "rear", "left", "right")
MEMBERS = {
    "episode_meta.json",
    "task_meta.json",
    "replay.json",
    "frames.jsonl",
    "events.jsonl",
    "rgb/meta.json",
    *(f"rgb/{cam}.mp4" for cam in CAMERAS),
}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate(directory: Path, row: dict) -> dict:
    import av

    meta = json.loads((directory / "episode_meta.json").read_text())
    replay = json.loads((directory / "replay.json").read_text())
    uid = row["episode_id"]
    if (
        meta.get("episode_id") != uid
        or meta.get("status") != "completed"
        or meta.get("sample_rate_hz") != 10
        or replay["source"]["episodeId"] != uid
        or replay["source"]["revision"] != row["source_revision"]
    ):
        raise ValueError(f"Replay identity/status/FPS mismatch: {uid}")
    count = 0
    with (directory / "frames.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            frame = json.loads(line)
            if frame.get("frame_index") != count:
                raise ValueError(f"Non-contiguous frame_index: {uid}, row {count}")
            for key in ("pose", *(f"camera_pose_{cam}" for cam in CAMERAS)):
                pose = frame.get(key)
                if (
                    not isinstance(pose, list)
                    or len(pose) != 6
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pose)
                ):
                    raise ValueError(f"Invalid {key}: {uid}, row {count}")
            count += 1
    if (
        count <= 0
        or meta["frame_count"] != count
        or replay["media"]["frameCount"] != count
        or replay["trajectory"]["frameCount"] != count
    ):
        raise ValueError(f"Frame/pose count mismatch: {uid}")
    for cam in CAMERAS:
        with av.open(str(directory / f"rgb/{cam}.mp4")) as container:
            stream = container.streams.video[0]
            if stream.frames != count or float(stream.average_rate or 0) != 10:
                raise ValueError(f"Video/pose count or FPS mismatch: {uid}/{cam}")
    return {
        "episode_id": uid,
        "episode_relative_path": f"episodes/{uid}",
        "status": "prepared",
        "source_revision": row["source_revision"],
        "pipeline_fingerprint": replay["pipeline"]["fingerprint"],
        "frame_count": count,
        "fps": 10,
        "scene_id": meta["scene_id"],
        "archive_size_bytes": row["archive_size_bytes"],
    }


def prepare_episode(archive_root: Path, output: Path, row: dict) -> dict:
    uid = str(uuid.UUID(row["episode_id"]))
    archive_path = archive_root / f"{uid}.tar.gz"
    if archive_path.stat().st_size != row["archive_size_bytes"]:
        raise ValueError(f"Archive size mismatch: {uid}")
    destination = output / "episodes" / uid
    marker = destination / ".prepared.json"
    if destination.exists():
        previous = json.loads(marker.read_text())
        if (
            previous["source_revision"] != row["source_revision"]
            or previous["archive_size_bytes"] != row["archive_size_bytes"]
        ):
            raise ValueError(f"Existing prepared source differs: {uid}")
        if any(not (destination / name).is_file() for name in MEMBERS):
            raise ValueError(f"Existing prepared episode is incomplete: {uid}")
        return previous
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{uid}.partial-{uuid.uuid4().hex[:8]}")
    partial.mkdir()
    seen: set[str] = set()
    with tarfile.open(archive_path, mode="r|gz", bufsize=1024 * 1024) as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member: {uid}/{name}")
            if name not in MEMBERS:
                continue
            if name in seen or not member.isfile():
                raise ValueError(f"Duplicate or non-file member: {uid}/{name}")
            seen.add(name)
            target = partial / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as handle:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
    if seen != MEMBERS:
        raise ValueError(f"Missing archive members: {uid}: {sorted(MEMBERS - seen)}")
    record = validate(partial, row)
    record["archive_path"] = str(archive_path)
    atomic_json(partial / ".prepared.json", record)
    os.rename(partial, destination)
    return record


def prepare(
    archive_root: Path, manifest: Path, output: Path, episode_ids: list[str] | None = None
) -> dict:
    archive_root, output = archive_root.resolve(), output.resolve()
    if output == archive_root or output.is_relative_to(archive_root):
        raise ValueError("Preparation output must be separate from the source archive directory")
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{manifest.resolve()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        query = (
            "SELECT episode_id,source_revision,archive_size_bytes,metadata_json,state FROM episodes"
        )
        parameters: list[str] = []
        if episode_ids:
            parameters = sorted({str(uuid.UUID(uid)) for uid in episode_ids})
            query += f" WHERE episode_id IN ({','.join('?' for _ in parameters)})"
        rows = [dict(row) for row in connection.execute(query + " ORDER BY episode_id", parameters)]
    if episode_ids and len(rows) != len(parameters):
        raise ValueError("Some selected UUIDs are missing from the source manifest")
    selection = [
        {k: row[k] for k in ("episode_id", "source_revision", "archive_size_bytes")} for row in rows
    ]
    selection_path = output / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError("Preparation selection changed; use a new work directory")
    else:
        atomic_json(selection_path, selection)
    prepared, errors = [], []
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"])
            if row["state"] != "downloaded" or metadata.get("validity") != "valid":
                raise ValueError("Source is not downloaded and valid")
            prepared.append(prepare_episode(archive_root, output, row))
        except Exception as exc:
            errors.append(
                {"episode_id": row["episode_id"], "error": f"{type(exc).__name__}: {exc}"}
            )
        print(
            f"准备进度={len(prepared) + len(errors)}/{len(rows)} 成功={len(prepared)} 失败={len(errors)}",
            flush=True,
        )
    report = {
        "total": len(rows),
        "prepared": len(prepared),
        "errors": errors,
        "total_frames": sum(row["frame_count"] for row in prepared),
    }
    atomic_json(output / "preparation_report.json", report)
    temporary = output / f".manifest-{uuid.uuid4().hex}.jsonl"
    with temporary.open("w") as handle:
        for row in prepared:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / "manifest.jsonl")
    if errors or not rows:
        raise ValueError(f"Preparation incomplete; see {output / 'preparation_report.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode-id", action="append")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.archive_root, args.manifest, args.output_root, args.episode_id),
            ensure_ascii=False,
            indent=2,
        )
    )
