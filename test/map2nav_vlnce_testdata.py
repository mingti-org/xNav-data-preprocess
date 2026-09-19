from __future__ import annotations

import gzip
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_video(path: Path, *, frame_count: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (16, 12),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot create the synthetic MP4 fixture")
    try:
        for index in range(frame_count):
            frame = np.full((12, 16, 3), 20 + index * 30, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def create_replay_source(root: Path, *, split: str = "train") -> Path:
    split_root = root / split
    scene_root = split_root / "scenes" / "TestScene"
    image_size = (20, 10)

    for relative, color in (
        ("level_0/layout.png", (10, 20, 30)),
        ("level_0/detail.png", (40, 50, 60)),
        ("graph_floor_0p000/graph.png", (70, 80, 90)),
    ):
        path = scene_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", image_size, color).save(path)

    bounds = {"min_x": 0.0, "min_z": -9.0, "max_x": 19.0, "max_z": 0.0}
    write_json(
        scene_root / "level_0" / "meta.json",
        {
            "scene_key": "TestScene",
            "level_id": 0,
            "coordinate_frame": "habitat_world_xz",
            "bounds": bounds,
            "scale_pixels_per_meter": 1.0,
            "width": image_size[0],
            "height": image_size[1],
            "projection": "canonical_pathfinder_bounds",
        },
    )
    write_json(
        scene_root / "graph_floor_0p000" / "meta.json",
        {
            "scene_key": "TestScene",
            "height": 0.0,
            "height_key": "0p000",
            "bounds": {"lower": [0.0, -1.0, -9.0], "upper": [19.0, 1.0, 0.0]},
            "shape": [image_size[1], image_size[0]],
            "meters_per_pixel": 1.0,
            "width": image_size[0],
            "height_pixels": image_size[1],
            "projection": "canonical_pathfinder_bounds",
            "scale_pixels_per_meter": 1.0,
        },
    )

    manifest: list[dict] = []
    specifications = [
        ("multi", [0, 1]),
        ("single_a", [0, 0]),
        ("single_b", [0, 0]),
    ]
    for trajectory_index, (name, levels) in enumerate(specifications):
        episode_dir_name = f"mp3d_TestScene_traj_{name}"
        episode_rel = Path("episodes") / split / episode_dir_name
        episode_dir = split_root / episode_rel
        overlays = episode_dir / "overlays"
        for filename, color in (
            ("trajectory_on_layout_level_0.png", (100, 110, 120)),
            ("trajectory_on_detail_level_0.png", (130, 140, 150)),
            ("trajectory_on_graph_floor_0p000.png", (160, 170, 180)),
        ):
            overlays.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", image_size, color).save(overlays / filename)
        Image.new("RGB", image_size, (1, 2, 3)).save(
            overlays / "trajectory_on_layout_level_0_backup.png"
        )

        for view in ("front", "back", "left", "right"):
            write_video(episode_dir / f"{view}.mp4")

        steps = [
            {
                "step_index": 0,
                "position": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "discrete_action_to_next": "move_forward",
                "discrete_action_to_next_id": 1,
                "video_frame_index": 0,
                "map_xy": [0, 9],
                "graph_xy": [0, 9],
                "floor_level_id": levels[0],
                "floorplan_xy": [0, 9],
            },
            {
                "step_index": 1,
                "position": [0.0, 0.0, -1.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "discrete_action_to_next": "stop",
                "discrete_action_to_next_id": 0,
                "video_frame_index": 1,
                "map_xy": [0, 8],
                "graph_xy": [0, 8],
                "floor_level_id": levels[1],
                "floorplan_xy": [0, 8],
            },
        ]
        write_jsonl(episode_dir / "steps.jsonl", steps)

        overlay_paths = [
            str(episode_rel / "overlays" / "trajectory_on_layout_level_0.png"),
            str(episode_rel / "overlays" / "trajectory_on_detail_level_0.png"),
            str(episode_rel / "overlays" / "trajectory_on_graph_floor_0p000.png"),
            str(episode_rel / "overlays" / "trajectory_on_layout_level_0_backup.png"),
        ]
        episode = {
            "dataset": "r2r",
            "role": None,
            "split": split,
            "episode_id": str(trajectory_index * 3),
            "episode_ids": [str(trajectory_index * 3 + offset) for offset in range(3)],
            "trajectory_id": name,
            "scene_id": "mp3d/TestScene/TestScene.glb",
            "scene_key": "TestScene",
            "instructions": [
                {
                    "episode_id": str(trajectory_index * 3 + offset),
                    "trajectory_id": name,
                    "instruction": f"instruction {offset} for {name}",
                }
                for offset in range(3)
            ],
            "num_steps": 2,
            "num_frames": 2,
            "video_hfov": 120.0,
            "video_views": ["front", "back", "left", "right"],
            "video_width": 16,
            "video_height": 12,
            "video_fps": 10,
            "scene_map_paths": {
                "levels": {
                    "0": {
                        "layout": "scenes/TestScene/level_0/layout.png",
                        "detail": "scenes/TestScene/level_0/detail.png",
                        "meta": "scenes/TestScene/level_0/meta.json",
                    }
                },
                "graph_floor": {
                    "height": 0.0,
                    "height_key": "0p000",
                    "directory": "scenes/TestScene/graph_floor_0p000",
                    "graph": "scenes/TestScene/graph_floor_0p000/graph.png",
                },
            },
            "overlay_paths": overlay_paths,
            "success": True,
            "map_projection": "canonical_pathfinder_bounds",
            "map_size": {"width": image_size[0], "height": image_size[1]},
        }
        write_json(episode_dir / "episode.json", episode)
        manifest.append(
            {
                "dataset": "r2r",
                "role": None,
                "split": split,
                "episode_id": episode["episode_id"],
                "episode_ids": episode["episode_ids"],
                "trajectory_id": name,
                "scene_id": episode["scene_id"],
                "scene_key": "TestScene",
                "episode_dir_name": episode_dir_name,
                "episode_dir": str(episode_rel),
                "video_hfov": 120.0,
                "video_views": ["front", "back", "left", "right"],
                "num_steps": 2,
                "num_frames": 2,
                "num_instructions": 3,
                "overlay_paths": overlay_paths,
            }
        )

    write_jsonl(split_root / "manifest.jsonl", manifest)
    write_jsonl(split_root / "errors.jsonl", [])
    return root


def create_rxr_replay_source(
    root: Path, *, split: str = "train"
) -> tuple[Path, Path]:
    source_root = create_replay_source(root, split=split)
    split_root = source_root / split
    manifest = read_jsonl(split_root / "manifest.jsonl")
    annotations: list[dict] = []
    next_episode_id = 1000

    for row in manifest:
        episode_path = split_root / row["episode_dir"] / "episode.json"
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        languages = (
            ["en-US", "en-IN", "hi-IN", "te-IN"]
            if episode["trajectory_id"] != "single_b"
            else ["hi-IN", "te-IN"]
        )
        instructions = []
        for language in languages:
            episode_id = str(next_episode_id)
            next_episode_id += 1
            text = f"{language} instruction for {episode['trajectory_id']}"
            instructions.append(
                {
                    "episode_id": episode_id,
                    "trajectory_id": episode["trajectory_id"],
                    "instruction": text,
                }
            )
            annotations.append(
                {
                    "episode_id": episode_id,
                    "trajectory_id": episode["trajectory_id"],
                    "scene_id": episode["scene_id"],
                    "info": {"role": "guide"},
                    "instruction": {
                        "instruction_text": text,
                        "language": language,
                    },
                }
            )

        episode.update(
            {
                "dataset": "rxr",
                "role": "guide",
                "episode_id": instructions[0]["episode_id"],
                "episode_ids": [item["episode_id"] for item in instructions],
                "instructions": instructions,
            }
        )
        write_json(episode_path, episode)
        row.update(
            {
                "dataset": "rxr",
                "role": "guide",
                "episode_id": episode["episode_id"],
                "episode_ids": episode["episode_ids"],
                "num_instructions": len(instructions),
            }
        )

    write_jsonl(split_root / "manifest.jsonl", manifest)
    annotation_path = root / f"{split}_guide.json.gz"
    with gzip.open(annotation_path, "wt", encoding="utf-8") as handle:
        json.dump({"episodes": annotations}, handle, ensure_ascii=False)
    return source_root, annotation_path


def create_scalevln_replay_source(
    root: Path, *, split: str = "train", instruction_prefix: str = "sc"
) -> Path:
    """ScaleVLN replay export: graph-only maps and one null-language instruction.

    Mirrors the real ScaleVLN raw contract: ``scene_map_paths.levels`` is empty,
    only the navmesh graph and its trajectory overlay are exported, every step
    carries ``floor_level_id == -1``, and instructions have no ``language``.
    """
    split_root = root / split
    scene_root = split_root / "scenes" / "TestScene"
    image_size = (20, 10)

    graph_directory = scene_root / "graph_floor_0p000"
    graph_path = graph_directory / "graph.png"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", image_size, (70, 80, 90)).save(graph_path)
    write_json(
        graph_directory / "meta.json",
        {
            "scene_key": "TestScene",
            "height": 0.0,
            "height_key": "0p000",
            "bounds": {"lower": [0.0, -1.0, -9.0], "upper": [19.0, 1.0, 0.0]},
            "shape": [image_size[1], image_size[0]],
            "meters_per_pixel": 1.0,
            "width": image_size[0],
            "height_pixels": image_size[1],
            "projection": "canonical_pathfinder_bounds",
            "scale_pixels_per_meter": 1.0,
        },
    )

    manifest: list[dict] = []
    for trajectory_index, name in enumerate(("scalevln_a", "scalevln_b")):
        episode_dir_name = f"hm3d_TestScene_traj_{name}"
        episode_rel = Path("episodes") / split / episode_dir_name
        episode_dir = split_root / episode_rel
        overlays = episode_dir / "overlays"
        overlays.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", image_size, (160, 170, 180)).save(
            overlays / "trajectory_on_graph_floor_0p000.png"
        )

        for view in ("front", "back", "left", "right"):
            write_video(episode_dir / f"{view}.mp4")

        steps = [
            {
                "step_index": 0,
                "position": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "discrete_action_to_next": "move_forward",
                "discrete_action_to_next_id": 1,
                "video_frame_index": 0,
                "map_xy": [0, 9],
                "graph_xy": [0, 9],
                "floor_level_id": -1,
                "floorplan_xy": [0, 9],
            },
            {
                "step_index": 1,
                "position": [0.0, 0.0, -1.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "discrete_action_to_next": "STOP",
                "discrete_action_to_next_id": 0,
                "video_frame_index": 1,
                "map_xy": [0, 8],
                "graph_xy": [0, 8],
                "floor_level_id": -1,
                "floorplan_xy": [0, 8],
            },
        ]
        write_jsonl(episode_dir / "steps.jsonl", steps)

        instruction_id = f"{instruction_prefix}_{trajectory_index}"
        overlay_paths = [
            str(episode_rel / "overlays" / "trajectory_on_graph_floor_0p000.png")
        ]
        episode = {
            "dataset": "scalevln",
            "role": None,
            "split": split,
            "episode_id": instruction_id,
            "episode_ids": [instruction_id],
            "trajectory_id": name,
            "scene_id": "hm3d/TestScene/TestScene.basis.glb",
            "scene_key": "TestScene",
            "instructions": [
                {
                    "episode_id": instruction_id,
                    "trajectory_id": name,
                    "instruction": f"instruction for {name}",
                    "language": None,
                }
            ],
            "num_steps": 2,
            "num_frames": 2,
            "video_hfov": 120.0,
            "video_views": ["front", "back", "left", "right"],
            "video_width": 16,
            "video_height": 12,
            "video_fps": 10,
            "scene_map_paths": {
                "levels": {},
                "graph_floor": {
                    "height": 0.0,
                    "height_key": "0p000",
                    "directory": "scenes/TestScene/graph_floor_0p000",
                    "graph": "scenes/TestScene/graph_floor_0p000/graph.png",
                },
            },
            "overlay_paths": overlay_paths,
            "success": True,
            "map_projection": "canonical_pathfinder_bounds",
            "map_size": {"width": image_size[0], "height": image_size[1]},
        }
        write_json(episode_dir / "episode.json", episode)
        manifest.append(
            {
                "dataset": "scalevln",
                "role": None,
                "split": split,
                "episode_id": instruction_id,
                "episode_ids": [instruction_id],
                "trajectory_id": name,
                "scene_id": episode["scene_id"],
                "scene_key": "TestScene",
                "episode_dir_name": episode_dir_name,
                "episode_dir": str(episode_rel),
                "video_hfov": 120.0,
                "video_views": ["front", "back", "left", "right"],
                "num_steps": 2,
                "num_frames": 2,
                "num_instructions": 1,
                "overlay_paths": overlay_paths,
            }
        )

    write_jsonl(split_root / "manifest.jsonl", manifest)
    write_jsonl(split_root / "errors.jsonl", [])
    return root
