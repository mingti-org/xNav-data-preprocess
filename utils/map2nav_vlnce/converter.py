"""Manifest-driven Map2Nav VLN-CE replay conversion."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from .assets import project_world_positions, resolve_map_bundle
from .coordinates import habitat_poses_to_xnav
from .filtering import FloorEligibility, SourceSchemaError, classify_floor_levels
from .schema import (
    RGB_VIEW_MAP,
    SCHEMA_VERSION,
    VideoInfo,
    build_features,
    build_modality,
    map_asset_keys,
    numeric_stats,
    write_episode_parquet,
)

RXR_ENGLISH_LANGUAGES = ("en-IN", "en-US")
EXPECTED_SOURCE_IDENTITIES = {
    "r2r": frozenset({("r2r", None)}),
    "rxr_guide": frozenset({("rxr", "guide"), ("rxr_en", "guide")}),
    "scalevln": frozenset({("scalevln", None)}),
}
SCALEVLN_FLOOR_ELIGIBILITY = FloorEligibility(
    accepted=True, source_level_id=None, visited_levels=(), reason=None
)


@dataclass(frozen=True)
class SourceInstruction:
    episode_id: str
    trajectory_id: str
    text: str
    language: str | None


@dataclass(frozen=True)
class SourceEpisode:
    manifest_index: int
    split_root: Path
    episode_dir: Path
    episode_dir_relative: str
    trajectory_id: str
    scene_key: str
    eligibility: FloorEligibility
    length: int
    source_instruction_count: int
    source_languages: tuple[str, ...]
    selected_instructions: tuple[SourceInstruction, ...]


@dataclass(frozen=True)
class ConversionEpisode:
    source: SourceEpisode
    instruction: SourceInstruction


def convert_dataset(
    input_root: str | Path,
    output_root: str | Path,
    dataset_name: str,
    split: str,
    max_episodes: int | None = None,
    chunk_size: int = 1000,
    resume: bool = False,
    overwrite: bool = False,
    num_workers: int = 1,
    skip_preflight: bool = False,
    rxr_annotations: str | Path | None = None,
    flat_output: bool = False,
) -> Path:
    """Convert a replay split, combining ScaleVLN shards when present."""

    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive when provided")
    if split not in {"train", "val_seen", "val_unseen"}:
        raise ValueError(f"unsupported split: {split!r}")
    if dataset_name not in {"r2r", "rxr_guide", "scalevln"}:
        raise ValueError(f"unsupported dataset_name: {dataset_name!r}")
    if dataset_name != "rxr_guide" and rxr_annotations is not None:
        raise ValueError("rxr_annotations is only valid for dataset_name='rxr_guide'")

    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    annotation_index: dict[str, dict[str, str]] | None = None
    annotation_context: dict[str, Any] | None = None
    if dataset_name == "rxr_guide":
        if rxr_annotations is None:
            raise SourceSchemaError(
                "rxr_guide conversion requires the authoritative RxR guide annotation "
                "JSON/JSON.GZ so language is selected by metadata rather than text heuristics"
            )
        annotation_path = Path(rxr_annotations).resolve()
        annotation_index = _load_rxr_annotations(annotation_path)
        annotation_context = _file_identity(annotation_path)
    split_roots = _source_split_roots(input_root, dataset_name=dataset_name, split=split)
    source_errors = [error for root in split_roots for error in _read_source_errors(root)]
    dataset_root = output_root if flat_output else output_root / split
    preexisting_output = dataset_root.exists()
    if preexisting_output and not overwrite and not resume:
        raise FileExistsError(
            f"output split already exists: {dataset_root}; use --resume or --overwrite"
        )
    context_path = dataset_root / "meta" / ".conversion" / "context.json"
    conversion_context = {
        "schema_version": SCHEMA_VERSION,
        "input_root": str(input_root),
        "dataset_name": dataset_name,
        "split": split,
        "chunk_size": chunk_size,
        "copy_mode": "copy2",
        "output_layout": "flat" if flat_output else "split_subdirectory",
        "episode_unit": "one_source_instruction",
        "rxr_languages": list(RXR_ENGLISH_LANGUAGES) if dataset_name == "rxr_guide" else None,
        "rxr_annotations": annotation_context,
    }
    if dataset_name == "scalevln":
        conversion_context["source_splits"] = [
            root.relative_to(input_root).as_posix() for root in split_roots
        ]
    if preexisting_output and resume and context_path.is_file():
        existing_context = _read_json(context_path)
        for key, value in conversion_context.items():
            if existing_context.get(key) != value:
                raise SourceSchemaError(
                    f"resume context mismatch for {key}: "
                    f"{existing_context.get(key)!r}, expected {value!r}"
                )
    elif preexisting_output and resume and any(dataset_root.iterdir()):
        raise SourceSchemaError(
            f"cannot resume non-empty output without conversion context: {dataset_root}"
        )

    scan_source = _scan_manifest_fast if skip_preflight else _scan_source
    candidates: list[SourceEpisode] = []
    for split_root in split_roots:
        shard_candidates = scan_source(
            split_root,
            dataset_name=dataset_name,
            split=split,
            annotation_index=annotation_index,
        )
        for candidate in shard_candidates:
            relative = candidate.episode_dir_relative
            if split_root != input_root / split:
                relative = (
                    split_root.relative_to(input_root) / relative
                ).as_posix()
            candidates.append(
                replace(
                    candidate,
                    manifest_index=len(candidates),
                    episode_dir_relative=relative,
                )
            )
    selected_source_ids = [
        instruction.episode_id
        for candidate in candidates
        for instruction in candidate.selected_instructions
    ]
    if len(selected_source_ids) != len(set(selected_source_ids)):
        raise SourceSchemaError("selected source instruction episode_ids are not globally unique")
    eligible_sources = [candidate for candidate in candidates if candidate.eligibility.accepted]
    eligible_episodes = [
        ConversionEpisode(source=candidate, instruction=instruction)
        for candidate in eligible_sources
        for instruction in candidate.selected_instructions
    ]
    selected = eligible_episodes[:max_episodes]
    skipped = [candidate for candidate in candidates if not candidate.eligibility.accepted]

    if preexisting_output and overwrite:
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    if not context_path.is_file():
        _write_json_atomic(context_path, conversion_context)
    journal_root = dataset_root / "meta" / ".conversion" / "episodes"
    journal_root.mkdir(parents=True, exist_ok=True)
    if resume:
        completed_indices = [
            int(path.stem.removeprefix("episode_"))
            for path in journal_root.glob("episode_*.json")
        ]
        if completed_indices and max(completed_indices) >= len(selected):
            raise SourceSchemaError(
                "max_episodes is smaller than the existing completed episode journal; "
                "resume may expand a conversion but cannot shrink it"
            )
    staging_root = dataset_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[int, ConversionEpisode, int]] = []
    global_frame_start = 0
    for episode_index, candidate in enumerate(selected):
        jobs.append((episode_index, candidate, global_frame_start))
        global_frame_start += candidate.source.length

    fragments: list[dict[str, Any]] = []
    expected_video: VideoInfo | None = None
    progress = tqdm(total=len(jobs), desc=f"Converting {dataset_name}/{split}", unit="episode")

    def process(job: tuple[int, ConversionEpisode, int]) -> tuple[dict[str, Any], bool]:
        episode_index, candidate, frame_start = job
        fragment_path = journal_root / f"episode_{episode_index:06d}.json"
        if resume and fragment_path.is_file():
            fragment = _read_json(fragment_path)
            _validate_resume_fragment(
                fragment,
                dataset_root=dataset_root,
                candidate=candidate,
                episode_index=episode_index,
                global_frame_start=frame_start,
            )
            return fragment, False
        _remove_episode_outputs(dataset_root, episode_index, chunk_size)
        fragment = _convert_episode(
            dataset_root=dataset_root,
            staging_root=staging_root,
            candidate=candidate,
            dataset_name=dataset_name,
            split=split,
            episode_index=episode_index,
            task_index=episode_index,
            global_frame_start=frame_start,
            chunk_size=chunk_size,
        )
        return fragment, True

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(process, job): job[0]
                for job in jobs
            }
            for future in as_completed(future_to_index):
                fragment, newly_converted = future.result()
                episode_index = int(fragment["episode_index"])
                if newly_converted:
                    fragment_path = journal_root / f"episode_{episode_index:06d}.json"
                    _write_json_atomic(fragment_path, fragment)

                video = VideoInfo(**fragment["video_info"])
                if expected_video is None:
                    expected_video = video
                elif video != expected_video:
                    raise SourceSchemaError(
                        f"video metadata differs across accepted episodes: "
                        f"{expected_video} vs {video}"
                    )
                fragments.append(fragment)
                progress.update(1)
        fragments.sort(key=lambda fragment: int(fragment["episode_index"]))
        for episode_index, fragment in enumerate(fragments):
            if int(fragment["episode_index"]) != episode_index:
                raise SourceSchemaError("conversion produced non-contiguous episode indices")
            if int(fragment["global_frame_start"]) != jobs[episode_index][2]:
                raise SourceSchemaError("conversion produced an incorrect global frame offset")
    except Exception as exc:
        _write_json_atomic(
            dataset_root / "meta" / "conversion_error.json",
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_name": dataset_name,
                "split": split,
                "accepted_before_error": len(fragments),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        progress.close()

    if expected_video is None:
        raise SourceSchemaError(f"no eligible episode selected for {dataset_name}/{split}")
    _write_final_metadata(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        split=split,
        chunk_size=chunk_size,
        video=expected_video,
        fragments=fragments,
        candidates=candidates,
        eligible_sources=eligible_sources,
        eligible_episodes=eligible_episodes,
        skipped=skipped,
        source_error_count=len(source_errors),
        max_episodes=max_episodes,
        num_workers=num_workers,
    )
    shutil.rmtree(staging_root, ignore_errors=True)
    error_path = dataset_root / "meta" / "conversion_error.json"
    if error_path.exists():
        error_path.unlink()
    return dataset_root


def _source_split_roots(
    input_root: Path, *, dataset_name: str, split: str
) -> list[Path]:
    direct = input_root / split
    if dataset_name != "scalevln":
        return [direct]
    shards = [root / split for root in sorted(input_root.glob("shard_*")) if root.is_dir()]
    if direct.exists() and shards:
        raise SourceSchemaError("ScaleVLN input contains both direct and sharded splits")
    return shards or [direct]


def _scan_source(
    split_root: Path,
    *,
    dataset_name: str,
    split: str,
    annotation_index: dict[str, dict[str, str]] | None,
) -> list[SourceEpisode]:
    manifest_path = split_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing replay manifest: {manifest_path}")
    manifest = _read_jsonl(manifest_path)
    if not manifest:
        raise SourceSchemaError(f"empty replay manifest: {manifest_path}")

    candidates: list[SourceEpisode] = []
    iterator = tqdm(manifest, desc=f"Preflight {dataset_name}/{split}", unit="episode")
    for manifest_index, row in enumerate(iterator):
        _validate_source_identity(row, dataset_name, manifest_index)
        if row.get("split") != split:
            raise SourceSchemaError(
                f"manifest row {manifest_index} split mismatch: {row.get('split')!r}"
            )
        relative = row.get("episode_dir")
        episode_dir = _safe_source_path(split_root, relative)
        episode = _read_json(episode_dir / "episode.json")
        steps = _read_jsonl(episode_dir / "steps.jsonl")
        _validate_episode_identity(row, episode, manifest_index)
        _validate_steps(steps, episode=episode, manifest=row, source_dir=episode_dir)
        source_instruction_count, source_languages, selected_instructions = (
            _select_source_instructions(
                row,
                episode,
                dataset_name=dataset_name,
                annotation_index=annotation_index,
                source_dir=episode_dir,
            )
        )
        eligibility = (
            SCALEVLN_FLOOR_ELIGIBILITY
            if dataset_name == "scalevln"
            else classify_floor_levels(steps)
        )
        candidates.append(
            SourceEpisode(
                manifest_index=manifest_index,
                split_root=split_root,
                episode_dir=episode_dir,
                episode_dir_relative=str(relative),
                trajectory_id=str(episode.get("trajectory_id", "")),
                scene_key=str(episode.get("scene_key", "")),
                eligibility=eligibility,
                length=len(steps),
                source_instruction_count=source_instruction_count,
                source_languages=source_languages,
                selected_instructions=selected_instructions,
            )
        )
    return candidates


def _read_source_errors(split_root: Path) -> list[dict[str, Any]]:
    errors_path = split_root / "errors.jsonl"
    if not errors_path.is_file():
        raise FileNotFoundError(f"missing replay errors file: {errors_path}")
    return _read_jsonl(errors_path)


def _scan_manifest_fast(
    split_root: Path,
    *,
    dataset_name: str,
    split: str,
    annotation_index: dict[str, dict[str, str]] | None,
) -> list[SourceEpisode]:
    """Build conversion jobs from manifest/episode metadata without step preflight.

    Conversion still reads and validates each selected steps file in
    ``_convert_episode``. This mode only avoids the separate full source scan;
    it is intended for trusted, already inspected replay roots.
    """
    manifest_path = split_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing replay manifest: {manifest_path}")
    manifest = _read_jsonl(manifest_path)
    if not manifest:
        raise SourceSchemaError(f"empty replay manifest: {manifest_path}")
    candidates: list[SourceEpisode] = []
    for manifest_index, row in enumerate(manifest):
        _validate_source_identity(row, dataset_name, manifest_index)
        if row.get("split") != split:
            raise SourceSchemaError(
                f"manifest row {manifest_index} split mismatch: {row.get('split')!r}"
            )
        relative = row.get("episode_dir")
        episode_dir = _safe_source_path(split_root, relative)
        episode = _read_json(episode_dir / "episode.json")
        _validate_episode_identity(row, episode, manifest_index)
        source_instruction_count, source_languages, selected_instructions = (
            _select_source_instructions(
                row,
                episode,
                dataset_name=dataset_name,
                annotation_index=annotation_index,
                source_dir=episode_dir,
            )
        )
        if dataset_name == "scalevln":
            eligibility = SCALEVLN_FLOOR_ELIGIBILITY
        else:
            overlay_paths = row.get("overlay_paths", [])
            level_ids = tuple(
                sorted(
                    {
                        int(match.group(1))
                        for path in overlay_paths
                        if (match := re.search(r"(?:layout|detail)_level_(\d+)", str(path)))
                    }
                )
            )
            if not level_ids:
                raise SourceSchemaError(f"manifest has no floor overlay levels: {episode_dir}")
            eligibility = FloorEligibility(
                accepted=len(level_ids) == 1,
                source_level_id=level_ids[0] if len(level_ids) == 1 else None,
                visited_levels=level_ids,
                reason=None if len(level_ids) == 1 else "multi_floor",
            )
        length = int(row.get("num_steps", 0))
        if length <= 0:
            raise SourceSchemaError(f"episode has invalid num_steps: {episode_dir}")
        candidates.append(
            SourceEpisode(
                manifest_index=manifest_index,
                split_root=split_root,
                episode_dir=episode_dir,
                episode_dir_relative=str(relative),
                trajectory_id=str(row.get("trajectory_id", "")),
                scene_key=str(row.get("scene_key", "")),
                eligibility=eligibility,
                length=length,
                source_instruction_count=source_instruction_count,
                source_languages=source_languages,
                selected_instructions=selected_instructions,
            )
        )
    return candidates


def _select_source_instructions(
    manifest: dict[str, Any],
    episode: dict[str, Any],
    *,
    dataset_name: str,
    annotation_index: dict[str, dict[str, str]] | None,
    source_dir: Path,
) -> tuple[int, tuple[str, ...], tuple[SourceInstruction, ...]]:
    raw_instructions = episode.get("instructions")
    if not isinstance(raw_instructions, list) or not raw_instructions:
        raise SourceSchemaError(f"instructions must be a non-empty list: {source_dir}")

    manifest_count = manifest.get("num_instructions")
    if int(manifest_count if manifest_count is not None else -1) != len(raw_instructions):
        raise SourceSchemaError(
            f"manifest num_instructions does not match episode.json: {source_dir}"
        )
    source_episode_ids = [str(value) for value in episode.get("episode_ids", [])]
    instruction_ids = [str(item.get("episode_id", "")) for item in raw_instructions]
    if source_episode_ids != instruction_ids:
        raise SourceSchemaError(
            f"episode_ids must exactly match instructions in source order: {source_dir}"
        )
    if len(instruction_ids) != len(set(instruction_ids)):
        raise SourceSchemaError(f"duplicate instruction episode_id: {source_dir}")

    selected: list[SourceInstruction] = []
    languages: set[str] = set()
    source_trajectory_id = str(episode.get("trajectory_id", ""))
    source_scene_id = str(episode.get("scene_id", ""))
    for index, item in enumerate(raw_instructions):
        if not isinstance(item, dict):
            raise SourceSchemaError(f"instruction {index} is not an object: {source_dir}")
        episode_id = str(item.get("episode_id", ""))
        trajectory_id = str(item.get("trajectory_id", ""))
        text = item.get("instruction")
        if not episode_id or not trajectory_id or not isinstance(text, str) or not text.strip():
            raise SourceSchemaError(f"invalid instruction {index}: {source_dir}")
        if trajectory_id != source_trajectory_id:
            raise SourceSchemaError(
                f"instruction {episode_id} trajectory_id mismatch: {source_dir}"
            )

        language: str | None = None
        if dataset_name == "rxr_guide":
            if annotation_index is None:
                raise SourceSchemaError("internal error: missing RxR annotation index")
            annotation = annotation_index.get(episode_id)
            if annotation is None:
                raise SourceSchemaError(
                    f"RxR annotation is missing replay episode_id={episode_id}: {source_dir}"
                )
            expected = {
                "trajectory_id": trajectory_id,
                "scene_id": source_scene_id,
                "instruction": text,
            }
            for key, value in expected.items():
                if annotation[key] != value:
                    raise SourceSchemaError(
                        f"RxR annotation mismatch for episode_id={episode_id} field={key}: "
                        f"{source_dir}"
                    )
            language = annotation["language"]
            languages.add(language)
        elif isinstance(item.get("language"), str):
            language = str(item["language"])
            languages.add(language)

        instruction = SourceInstruction(
            episode_id=episode_id,
            trajectory_id=trajectory_id,
            text=text,
            language=language,
        )
        if dataset_name in {"r2r", "scalevln"} or language in RXR_ENGLISH_LANGUAGES:
            selected.append(instruction)

    return len(raw_instructions), tuple(sorted(languages)), tuple(selected)


def _convert_episode(
    *,
    dataset_root: Path,
    staging_root: Path,
    candidate: ConversionEpisode,
    dataset_name: str,
    split: str,
    episode_index: int,
    task_index: int,
    global_frame_start: int,
    chunk_size: int,
) -> dict[str, Any]:
    source_episode = candidate.source
    instruction = candidate.instruction
    episode = _read_json(source_episode.episode_dir / "episode.json")
    steps = _read_jsonl(source_episode.episode_dir / "steps.jsonl")
    source_level_id = None
    if dataset_name != "scalevln":
        eligibility = classify_floor_levels(steps)
        if not eligibility.accepted or eligibility.source_level_id is None:
            raise SourceSchemaError(
                f"accepted source episode changed floor eligibility: {source_episode.episode_dir}"
            )
        source_level_id = eligibility.source_level_id
    _validate_steps(steps, episode=episode, manifest=None, source_dir=source_episode.episode_dir)
    _validate_selected_instruction(episode, instruction, source_dir=source_episode.episode_dir)
    bundle = resolve_map_bundle(
        source_episode.split_root,
        episode,
        source_level_id,
        dataset_name=dataset_name,
    )
    positions = np.asarray([step["position"] for step in steps], dtype=np.float64)
    rotations = np.asarray([step["rotation"] for step in steps], dtype=np.float64)
    source_pixels = np.asarray([step["floorplan_xy"] for step in steps], dtype=np.int32)
    projected = project_world_positions(positions, bundle.projection, rounded=True)
    max_projection_error = int(np.abs(projected - source_pixels).max(initial=0))
    if max_projection_error > 1:
        raise SourceSchemaError(
            f"floorplan projection differs by {max_projection_error}px: {source_episode.episode_dir}"
        )

    video = _video_info(episode, source_episode.episode_dir)
    states = habitat_poses_to_xnav(positions, rotations)
    rows, stats = _build_rows(
        steps=steps,
        states=states,
        positions=positions,
        rotations=rotations,
        episode_index=episode_index,
        task_index=task_index,
        global_frame_start=global_frame_start,
        fps=video.fps,
    )

    chunk = episode_index // chunk_size
    episode_name = f"episode_{episode_index:06d}"
    stage = staging_root / episode_name
    shutil.rmtree(stage, ignore_errors=True)
    staged_files: dict[str, Path] = {}
    parquet_relative = Path("data") / f"chunk-{chunk:03d}" / f"{episode_name}.parquet"
    stage.mkdir(parents=True, exist_ok=True)
    parquet_stage = stage / "episode.parquet"
    write_episode_parquet(parquet_stage, rows)
    staged_files[parquet_relative.as_posix()] = parquet_stage

    for source_view, output_view in RGB_VIEW_MAP.items():
        source_path = source_episode.episode_dir / f"{source_view}.mp4"
        relative = (
            Path("videos")
            / f"chunk-{chunk:03d}"
            / f"video.{output_view}"
            / f"{episode_name}.mp4"
        )
        stage_path = stage / f"video.{output_view}.mp4"
        _copy_file(source_path, stage_path)
        staged_files[relative.as_posix()] = stage_path

    map_assets: dict[str, str] = {}
    for key in map_asset_keys(dataset_name):
        relative = Path("maps") / f"chunk-{chunk:03d}" / episode_name / f"{key}.png"
        stage_path = stage / f"map.{key}.png"
        _copy_file(bundle.sources[key], stage_path)
        staged_files[relative.as_posix()] = stage_path
        map_assets[key] = relative.as_posix()

    _validate_staged_episode(staged_files, expected_rows=len(rows))
    for parent in {str((dataset_root / relative).parent) for relative in staged_files}:
        Path(parent).mkdir(parents=True, exist_ok=True)
    for relative, source_path in staged_files.items():
        target = dataset_root / relative
        os.replace(source_path, target)
    stage.rmdir()

    instruction_payload: dict[str, Any] = {
        "episode_id": instruction.episode_id,
        "trajectory_id": instruction.trajectory_id,
        "instruction": instruction.text,
    }
    if instruction.language is not None:
        instruction_payload["language"] = instruction.language
    extras = {
        "schema_version": SCHEMA_VERSION,
        "episode_index": episode_index,
        "dataset_name": dataset_name,
        "role": episode.get("role"),
        "split": split,
        "trajectory_id": str(episode.get("trajectory_id", "")),
        "scene_key": str(episode.get("scene_key", "")),
        "source_episode_ids": [str(value) for value in episode.get("episode_ids", [])],
        "source_episode_dir": source_episode.episode_dir_relative,
        "instructions": [instruction_payload],
        "video": {
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "hfov": video.hfov,
            "source_view_order": list(episode.get("video_views", [])),
        },
        "map_size": [bundle.height, bundle.width],
        "map_projection": bundle.projection,
        "map_assets": map_assets,
    }
    return {
        "source_manifest_index": source_episode.manifest_index,
        "source_episode_dir": source_episode.episode_dir_relative,
        "source_instruction_episode_id": instruction.episode_id,
        "episode_index": episode_index,
        "global_frame_start": global_frame_start,
        "length": len(rows),
        "video_info": {
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "hfov": video.hfov,
        },
        "output_files": list(staged_files),
        "episode": {
            "episode_index": episode_index,
            "tasks": [instruction.text],
            "length": len(rows),
        },
        "task": {"task_index": task_index, "task": instruction.text},
        "stats": {"episode_index": episode_index, "stats": stats},
        "extras": extras,
    }


def _build_rows(
    *,
    steps: list[dict[str, Any]],
    states: np.ndarray,
    positions: np.ndarray,
    rotations: np.ndarray,
    episode_index: int,
    task_index: int,
    global_frame_start: int,
    fps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_positions = positions.astype(np.float32)
    raw_rotations = rotations.astype(np.float32)
    floorplan = np.asarray([step["floorplan_xy"] for step in steps], dtype=np.int32)
    discrete = np.asarray(
        [int(step["discrete_action_to_next_id"]) for step in steps], dtype=np.int32
    ).reshape(-1, 1)
    rows: list[dict[str, Any]] = []
    for frame_index in range(len(steps)):
        state = states[frame_index].tolist()
        rows.append(
            {
                "annotation.human.action.task_description": [task_index],
                "observation.state": state,
                "action": list(state),
                "frame_index": frame_index,
                "timestamp": np.float32(frame_index / fps).item(),
                "index": global_frame_start + frame_index,
                "episode_index": episode_index,
                "task_index": task_index,
                "extra.habitat_world_position": raw_positions[frame_index].tolist(),
                "extra.habitat_world_rotation_xyzw": raw_rotations[frame_index].tolist(),
                "extra.floorplan_xy": floorplan[frame_index].tolist(),
                "extra.discrete_action_to_next_id": discrete[frame_index].tolist(),
                "extra.cot": "",
            }
        )
    stats = numeric_stats(
        {
            "observation.state": states,
            "action": states,
            "extra.habitat_world_position": raw_positions,
            "extra.habitat_world_rotation_xyzw": raw_rotations,
            "extra.floorplan_xy": floorplan,
            "extra.discrete_action_to_next_id": discrete,
        }
    )
    return rows, stats


def _write_final_metadata(
    *,
    dataset_root: Path,
    dataset_name: str,
    split: str,
    chunk_size: int,
    video: VideoInfo,
    fragments: list[dict[str, Any]],
    candidates: list[SourceEpisode],
    eligible_sources: list[SourceEpisode],
    eligible_episodes: list[ConversionEpisode],
    skipped: list[SourceEpisode],
    source_error_count: int,
    max_episodes: int | None,
    num_workers: int,
) -> None:
    meta = dataset_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    total_episodes = len(fragments)
    total_frames = sum(int(fragment["length"]) for fragment in fragments)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "map2nav_vlnce",
        "schema_version": SCHEMA_VERSION,
        "fps": video.fps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_episodes,
        "total_videos": total_episodes * len(RGB_VIEW_MAP),
        "total_chunks": math.ceil(total_episodes / chunk_size),
        "chunks_size": chunk_size,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": build_features(video),
        "splits": {split: f"0:{total_episodes}"},
    }
    _write_json_atomic(meta / "info.json", info)
    _write_jsonl_atomic(meta / "tasks.jsonl", [fragment["task"] for fragment in fragments])
    _write_jsonl_atomic(meta / "episodes.jsonl", [fragment["episode"] for fragment in fragments])
    _write_jsonl_atomic(meta / "episodes_stats.jsonl", [fragment["stats"] for fragment in fragments])
    _write_jsonl_atomic(meta / "episodes_extras.jsonl", [fragment["extras"] for fragment in fragments])
    _write_jsonl_atomic(
        meta / "skipped_episodes.jsonl",
        [
            {
                "source_manifest_index": candidate.manifest_index,
                "source_episode_dir": candidate.episode_dir_relative,
                "trajectory_id": candidate.trajectory_id,
                "scene_key": candidate.scene_key,
                "reason": "multi_floor",
                "visited_levels": list(candidate.eligibility.visited_levels),
                "source_instruction_count": candidate.source_instruction_count,
                "selected_instruction_count": len(candidate.selected_instructions),
            }
            for candidate in skipped
        ]
        + [
            {
                "source_manifest_index": candidate.manifest_index,
                "source_episode_dir": candidate.episode_dir_relative,
                "trajectory_id": candidate.trajectory_id,
                "scene_key": candidate.scene_key,
                "reason": "no_selected_instruction",
                "visited_levels": list(candidate.eligibility.visited_levels),
                "source_instruction_count": candidate.source_instruction_count,
                "selected_instruction_count": 0,
                "source_languages": list(candidate.source_languages),
            }
            for candidate in eligible_sources
            if not candidate.selected_instructions
        ],
    )
    selected_sources = [candidate for candidate in eligible_sources if candidate.selected_instructions]
    eligible_language_counts = Counter(
        episode.instruction.language
        for episode in eligible_episodes
        if episode.instruction.language is not None
    )
    accepted_language_counts = Counter(
        fragment["extras"]["instructions"][0].get("language")
        for fragment in fragments
        if fragment["extras"]["instructions"][0].get("language") is not None
    )
    floor_report = (
        {
            "floor_filter": "not_applied",
            "floor_filter_reason": "source_has_no_floor_metadata",
            "eligible_source_episodes": len(eligible_sources),
        }
        if dataset_name == "scalevln"
        else {
            "eligible_single_floor": len(eligible_sources),
            "eligible_single_floor_with_selected_instructions": len(selected_sources),
            "language_filtered_single_floor": len(eligible_sources) - len(selected_sources),
            "skipped_multi_floor": len(skipped),
            "skipped_multi_floor_selected_instructions": sum(
                len(candidate.selected_instructions) for candidate in skipped
            ),
        }
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "split": split,
        "copy_mode": "copy2",
        "episode_unit": "one_source_instruction",
        "rxr_languages": list(RXR_ENGLISH_LANGUAGES) if dataset_name == "rxr_guide" else None,
        "source_manifest_total": len(candidates),
        "source_recorded_errors": source_error_count,
        "source_instruction_total": sum(
            candidate.source_instruction_count for candidate in candidates
        ),
        "selected_instruction_total_before_floor_filter": sum(
            len(candidate.selected_instructions) for candidate in candidates
        ),
        **floor_report,
        "eligible_instruction_episodes": len(eligible_episodes),
        "eligible_instruction_language_counts": dict(sorted(eligible_language_counts.items())),
        "accepted": total_episodes,
        "accepted_frames": total_frames,
        "accepted_instruction_language_counts": dict(sorted(accepted_language_counts.items())),
        "unconverted_eligible_instruction_episodes_due_to_limit": (
            len(eligible_episodes) - total_episodes
        ),
        "errors": 0,
        "max_episodes": max_episodes,
        "num_workers": num_workers,
        "complete_success_manifest_conversion": max_episodes is None,
        "complete_source_conversion": max_episodes is None and source_error_count == 0,
    }
    _write_json_atomic(meta / "conversion_report.json", report)
    _write_json_atomic(meta / "modality.json", build_modality())


def _validate_steps(
    steps: list[dict[str, Any]],
    *,
    episode: dict[str, Any],
    manifest: dict[str, Any] | None,
    source_dir: Path,
) -> None:
    if not steps:
        raise SourceSchemaError(f"empty steps.jsonl: {source_dir}")
    expected = len(steps)
    for source, label in ((episode, "episode.json"), (manifest, "manifest")):
        if source is None:
            continue
        for key in ("num_steps", "num_frames"):
            if int(source.get(key, -1)) != expected:
                raise SourceSchemaError(
                    f"{label} {key} does not match steps ({expected}): {source_dir}"
                )
    for index, step in enumerate(steps):
        if int(step.get("step_index", -1)) != index:
            raise SourceSchemaError(f"non-contiguous step_index at {source_dir} frame {index}")
        if int(step.get("video_frame_index", -1)) != index:
            raise SourceSchemaError(f"video_frame_index mismatch at {source_dir} frame {index}")
        position = np.asarray(step.get("position"), dtype=np.float64)
        rotation = np.asarray(step.get("rotation"), dtype=np.float64)
        floorplan = np.asarray(step.get("floorplan_xy"))
        if position.shape != (3,) or rotation.shape != (4,) or floorplan.shape != (2,):
            raise SourceSchemaError(f"invalid pose/floorplan shape at {source_dir} frame {index}")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
            raise SourceSchemaError(f"non-finite pose at {source_dir} frame {index}")
        if np.linalg.norm(rotation) <= 0.0:
            raise SourceSchemaError(f"zero-norm quaternion at {source_dir} frame {index}")
        if step.get("map_xy") != step.get("graph_xy") or step.get("map_xy") != step.get(
            "floorplan_xy"
        ):
            raise SourceSchemaError(f"unaligned map coordinates at {source_dir} frame {index}")
        action_id = int(step.get("discrete_action_to_next_id", -1))
        if action_id not in {0, 1, 2, 3}:
            raise SourceSchemaError(f"invalid discrete action at {source_dir} frame {index}")
    if int(steps[-1].get("discrete_action_to_next_id", -1)) != 0:
        raise SourceSchemaError(f"terminal frame is not STOP: {source_dir}")


def _validate_episode_identity(
    manifest: dict[str, Any], episode: dict[str, Any], manifest_index: int
) -> None:
    for key in ("dataset", "role", "split", "trajectory_id", "scene_key", "episode_ids"):
        if manifest.get(key) != episode.get(key):
            raise SourceSchemaError(
                f"manifest row {manifest_index} disagrees with episode.json for {key!r}"
            )


def _validate_source_identity(
    manifest: dict[str, Any], dataset_name: str, manifest_index: int
) -> None:
    identity = (manifest.get("dataset"), manifest.get("role"))
    if identity not in EXPECTED_SOURCE_IDENTITIES[dataset_name]:
        raise SourceSchemaError(
            f"manifest row {manifest_index} dataset/role mismatch: "
            f"{identity[0]!r}/{identity[1]!r}"
        )


def _validate_selected_instruction(
    episode: dict[str, Any], instruction: SourceInstruction, *, source_dir: Path
) -> None:
    matches = [
        item
        for item in episode.get("instructions", [])
        if isinstance(item, dict) and str(item.get("episode_id", "")) == instruction.episode_id
    ]
    if len(matches) != 1:
        raise SourceSchemaError(
            f"selected instruction episode_id={instruction.episode_id} changed: {source_dir}"
        )
    item = matches[0]
    if (
        str(item.get("trajectory_id", "")) != instruction.trajectory_id
        or item.get("instruction") != instruction.text
    ):
        raise SourceSchemaError(
            f"selected instruction episode_id={instruction.episode_id} content changed: "
            f"{source_dir}"
        )


def _video_info(episode: dict[str, Any], source_dir: Path) -> VideoInfo:
    try:
        video = VideoInfo(
            width=int(episode["video_width"]),
            height=int(episode["video_height"]),
            fps=int(episode["video_fps"]),
            hfov=float(episode["video_hfov"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceSchemaError(f"invalid video metadata: {source_dir}") from exc
    if video.width <= 0 or video.height <= 0 or video.fps <= 0:
        raise SourceSchemaError(f"invalid video dimensions/fps: {source_dir}")
    return video


def _copy_file(source: Path, target: Path) -> None:
    if not source.is_file() or source.stat().st_size <= 0:
        raise SourceSchemaError(f"missing or empty source asset: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if target.stat().st_size != source.stat().st_size:
        raise OSError(f"copied file size mismatch: {source} -> {target}")


def _validate_staged_episode(staged_files: dict[str, Path], *, expected_rows: int) -> None:
    for path in staged_files.values():
        if not path.is_file() or path.stat().st_size <= 0:
            raise OSError(f"staged output is missing or empty: {path}")
    parquet = next(
        path for relative, path in staged_files.items() if relative.endswith(".parquet")
    )
    if pq.read_metadata(parquet).num_rows != expected_rows:
        raise OSError(f"staged parquet row count mismatch: {parquet}")


def _validate_resume_fragment(
    fragment: dict[str, Any],
    *,
    dataset_root: Path,
    candidate: ConversionEpisode,
    episode_index: int,
    global_frame_start: int,
) -> None:
    source = candidate.source
    expected = {
        "source_manifest_index": source.manifest_index,
        "source_episode_dir": source.episode_dir_relative,
        "source_instruction_episode_id": candidate.instruction.episode_id,
        "episode_index": episode_index,
        "global_frame_start": global_frame_start,
        "length": source.length,
    }
    for key, value in expected.items():
        if fragment.get(key) != value:
            raise SourceSchemaError(
                f"resume journal mismatch for episode {episode_index}: {key}={fragment.get(key)!r}, "
                f"expected {value!r}"
            )
    for relative in fragment.get("output_files", []):
        path = dataset_root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise SourceSchemaError(f"resume output missing or empty: {path}")


def _remove_episode_outputs(dataset_root: Path, episode_index: int, chunk_size: int) -> None:
    chunk = episode_index // chunk_size
    name = f"episode_{episode_index:06d}"
    paths = [dataset_root / "data" / f"chunk-{chunk:03d}" / f"{name}.parquet"]
    paths.extend(
        dataset_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / f"video.{view}"
        / f"{name}.mp4"
        for view in RGB_VIEW_MAP.values()
    )
    paths.append(dataset_root / "maps" / f"chunk-{chunk:03d}" / name)
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _safe_source_path(root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise SourceSchemaError(f"invalid episode_dir in manifest: {raw!r}")
    resolved_root = root.resolve()
    path = (resolved_root / raw).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise SourceSchemaError(f"episode_dir escapes split root: {raw!r}") from exc
    if not path.is_dir():
        raise SourceSchemaError(f"missing episode directory: {path}")
    return path


def _load_rxr_annotations(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing RxR annotation file: {path}")
    try:
        opener = gzip.open if path.suffix == ".gz" else Path.open
        if path.suffix == ".gz":
            with opener(path, "rt", encoding="utf-8") as handle:
                value = json.load(handle)
        else:
            with opener(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
    except Exception as exc:
        raise SourceSchemaError(f"cannot read RxR annotation JSON: {path}") from exc
    episodes = value.get("episodes") if isinstance(value, dict) else None
    if not isinstance(episodes, list) or not episodes:
        raise SourceSchemaError(f"RxR annotation JSON has no episodes: {path}")

    index: dict[str, dict[str, str]] = {}
    for row_index, episode in enumerate(episodes):
        instruction = episode.get("instruction") if isinstance(episode, dict) else None
        if not isinstance(instruction, dict):
            raise SourceSchemaError(f"invalid RxR annotation episode {row_index}: {path}")
        episode_id = str(episode.get("episode_id", ""))
        record = {
            "trajectory_id": str(episode.get("trajectory_id", "")),
            "scene_id": str(episode.get("scene_id", "")),
            "instruction": instruction.get("instruction_text"),
            "language": instruction.get("language"),
        }
        if (
            not episode_id
            or not record["trajectory_id"]
            or not record["scene_id"]
            or not isinstance(record["instruction"], str)
            or not record["instruction"].strip()
            or not isinstance(record["language"], str)
            or not record["language"]
        ):
            raise SourceSchemaError(f"invalid RxR annotation episode {row_index}: {path}")
        if episode_id in index:
            raise SourceSchemaError(f"duplicate RxR annotation episode_id={episode_id}: {path}")
        index[episode_id] = record
    return index


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SourceSchemaError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise SourceSchemaError(f"JSON value must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SourceSchemaError(f"JSONL row is not an object: {path}:{line_number}")
                rows.append(value)
    except SourceSchemaError:
        raise
    except Exception as exc:
        raise SourceSchemaError(f"cannot read JSONL: {path}") from exc
    return rows


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)
