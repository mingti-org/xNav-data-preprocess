"""Stable Map2Nav VLN-CE output schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "map2nav_vlnce_v2"
POSE_AXES = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
MAP_ASSET_KEYS = (
    "graph",
    "graph_overlay",
    "floorplan",
    "floorplan_overlay",
    "floorplan_detail",
    "floorplan_detail_overlay",
)
# ScaleVLN replay only exports the navmesh graph and its trajectory overlay, so
# its map contract is a strict subset of the Map2Nav floorplan asset set.
SCALEVLN_MAP_ASSET_KEYS = (
    "graph",
    "graph_overlay",
)


def map_asset_keys(dataset_name: str) -> tuple[str, ...]:
    """Return the required map asset keys for one source dataset."""
    if dataset_name == "scalevln":
        return SCALEVLN_MAP_ASSET_KEYS
    return MAP_ASSET_KEYS


RGB_VIEW_MAP = {
    "front": "front",
    "left": "left",
    "right": "right",
    "back": "rear",
}
PARQUET_COLUMNS = [
    "annotation.human.action.task_description",
    "observation.state",
    "action",
    "frame_index",
    "timestamp",
    "index",
    "episode_index",
    "task_index",
    "extra.habitat_world_position",
    "extra.habitat_world_rotation_xyzw",
    "extra.floorplan_xy",
    "extra.discrete_action_to_next_id",
    "extra.cot",
]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: int
    hfov: float


def parquet_schema() -> pa.Schema:
    fixed = pa.list_
    return pa.schema(
        [
            pa.field(PARQUET_COLUMNS[0], fixed(pa.int32(), 1)),
            pa.field(PARQUET_COLUMNS[1], fixed(pa.float32(), 7)),
            pa.field(PARQUET_COLUMNS[2], fixed(pa.float32(), 7)),
            pa.field("frame_index", pa.int64()),
            pa.field("timestamp", pa.float32()),
            pa.field("index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("extra.habitat_world_position", fixed(pa.float32(), 3)),
            pa.field("extra.habitat_world_rotation_xyzw", fixed(pa.float32(), 4)),
            pa.field("extra.floorplan_xy", fixed(pa.int32(), 2)),
            pa.field("extra.discrete_action_to_next_id", fixed(pa.int32(), 1)),
            pa.field("extra.cot", pa.string()),
        ]
    )


def write_episode_parquet(path: Any, rows: list[dict[str, Any]]) -> None:
    columns = {key: [row[key] for row in rows] for key in PARQUET_COLUMNS}
    table = pa.Table.from_pydict(columns, schema=parquet_schema())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def build_features(video: VideoInfo) -> dict[str, Any]:
    def video_feature() -> dict[str, Any]:
        return {
            "dtype": "video",
            "shape": [video.height, video.width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": video.height,
                "video.width": video.width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": video.fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    return {
        "annotation.human.action.task_description": {
            "dtype": "int32",
            "shape": [1],
            "names": None,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": {"axes": POSE_AXES},
        },
        "video.front": video_feature(),
        "video.left": video_feature(),
        "video.right": video_feature(),
        "video.rear": video_feature(),
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": {"axes": POSE_AXES},
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "extra.habitat_world_position": {
            "dtype": "float32",
            "shape": [3],
            "names": {"axes": ["x", "y", "z"]},
        },
        "extra.habitat_world_rotation_xyzw": {
            "dtype": "float32",
            "shape": [4],
            "names": {"axes": ["qx", "qy", "qz", "qw"]},
        },
        "extra.floorplan_xy": {
            "dtype": "int32",
            "shape": [2],
            "names": {"axes": ["u", "v"]},
        },
        "extra.discrete_action_to_next_id": {
            "dtype": "int32",
            "shape": [1],
            "names": None,
        },
        "extra.cot": {"dtype": "string", "shape": [1], "names": None},
    }


def build_modality() -> dict[str, Any]:
    return {
        "state": {
            "drone": {
                "start": 0,
                "end": 7,
                "absolute": True,
                "rotation_type": "quaternion",
                "original_key": "observation.state",
            }
        },
        "action": {
            "pose": {
                "start": 0,
                "end": 7,
                "absolute": True,
                "rotation_type": "quaternion",
                "original_key": "action",
            }
        },
        "video": {
            "front": {"original_key": "video.front"},
            "left": {"original_key": "video.left"},
            "right": {"original_key": "video.right"},
            "rear": {"original_key": "video.rear"},
        },
        "annotation": {"human.action.task_description": {}},
    }


def numeric_stats(values: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for key, value in values.items():
        array = np.asarray(value)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        stats[key] = {
            "min": array.min(axis=0).tolist(),
            "max": array.max(axis=0).tolist(),
            "mean": array.mean(axis=0).tolist(),
            "std": array.std(axis=0).tolist(),
            "count": [int(array.shape[0])],
        }
    return stats
