"""Resolve and validate the episode-bound map assets.

Map2Nav replay exports six floorplan/graph assets; ScaleVLN replay exports only
the navmesh graph plus its trajectory overlay. Both are validated against the
same canonical-pathfinder-bounds projection, taken from the level metadata for
Map2Nav and from the graph-floor metadata for ScaleVLN.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .filtering import SourceSchemaError
from .schema import map_asset_keys


@dataclass(frozen=True)
class MapBundle:
    width: int
    height: int
    sources: dict[str, Path]
    projection: dict[str, Any]


def resolve_map_bundle(
    split_root: Path,
    episode: dict[str, Any],
    source_level_id: int | None,
    *,
    dataset_name: str,
) -> MapBundle:
    scene_paths = _mapping(episode.get("scene_map_paths"), "scene_map_paths")
    graph_floor = _mapping(scene_paths.get("graph_floor"), "scene_map_paths.graph_floor")
    height_key = str(graph_floor.get("height_key", ""))
    if not height_key:
        raise SourceSchemaError("scene_map_paths.graph_floor.height_key is missing")

    overlay_paths = [str(value) for value in episode.get("overlay_paths", [])]
    graph_sources = {
        "graph": _source_path(split_root, graph_floor.get("graph")),
        "graph_overlay": _exact_overlay(
            split_root,
            overlay_paths,
            rf"trajectory_on_graph_floor_{re.escape(height_key)}\.png",
        ),
    }
    level_meta: dict[str, Any] | None = None
    if dataset_name == "scalevln":
        sources = graph_sources
    else:
        levels = _mapping(scene_paths.get("levels"), "scene_map_paths.levels")
        level = _mapping(
            levels.get(str(source_level_id)),
            f"scene_map_paths.levels[{source_level_id}]",
        )
        sources = {
            **graph_sources,
            "floorplan": _source_path(split_root, level.get("layout")),
            "floorplan_overlay": _exact_overlay(
                split_root,
                overlay_paths,
                rf"trajectory_on_layout_level_{source_level_id}\.png",
            ),
            "floorplan_detail": _source_path(split_root, level.get("detail")),
            "floorplan_detail_overlay": _exact_overlay(
                split_root,
                overlay_paths,
                rf"trajectory_on_detail_level_{source_level_id}\.png",
            ),
        }
        level_meta = _read_json(_source_path(split_root, level.get("meta")))
    if tuple(sources) != map_asset_keys(dataset_name):
        raise AssertionError("map asset key order drifted from the stable schema")

    dimensions: dict[str, tuple[int, int]] = {}
    for key, path in sources.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise SourceSchemaError(f"missing or empty {key} map asset: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions[key] = image.size
        except Exception as exc:
            raise SourceSchemaError(f"cannot decode {key} map asset: {path}") from exc
    unique_sizes = set(dimensions.values())
    if len(unique_sizes) != 1:
        raise SourceSchemaError(f"map asset sizes differ: {dimensions}")
    width, height = unique_sizes.pop()

    graph_directory = _source_path(split_root, graph_floor.get("directory"))
    graph_meta = _read_json(graph_directory / "meta.json")
    projection = (
        _build_graph_projection(graph_meta, width=width, height=height)
        if level_meta is None
        else _build_projection(level_meta, graph_meta, width=width, height=height)
    )
    return MapBundle(width=width, height=height, sources=sources, projection=projection)


def project_world_positions(
    positions: np.ndarray,
    projection: dict[str, Any],
    *,
    rounded: bool,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape [N,3], got {positions.shape}")
    matrix = np.asarray(projection["world_xz_to_pixel"], dtype=np.float64)
    homogeneous = np.stack(
        [positions[:, 0], positions[:, 2], np.ones(len(positions), dtype=np.float64)]
    )
    pixels = (matrix @ homogeneous)[:2].T
    if not rounded:
        return pixels
    pixels = np.rint(pixels)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, int(projection["width"]) - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, int(projection["height"]) - 1)
    return pixels.astype(np.int32)


def _graph_bounds_xz(
    graph_meta: dict[str, Any], *, width: int, height: int
) -> tuple[float, float, float, float]:
    """Validate graph-floor metadata and return its (min_x, min_z, max_x, max_z)."""
    graph_shape = graph_meta.get("shape")
    if list(graph_shape or []) != [height, width]:
        raise SourceSchemaError(
            f"graph meta shape {graph_shape!r} does not match map images {[height, width]}"
        )
    graph_bounds = _mapping(graph_meta.get("bounds"), "graph meta bounds")
    lower = np.asarray(graph_bounds.get("lower"), dtype=np.float64)
    upper = np.asarray(graph_bounds.get("upper"), dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,):
        raise SourceSchemaError("graph meta bounds must contain 3D lower/upper vectors")
    min_x = float(lower[0])
    min_z = float(lower[2])
    max_x = float(upper[0])
    max_z = float(upper[2])
    if max_x <= min_x or max_z <= min_z:
        raise SourceSchemaError(f"invalid graph bounds: {graph_bounds}")
    return min_x, min_z, max_x, max_z


def _build_graph_projection(
    graph_meta: dict[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Projection for sources that export only the navmesh graph (ScaleVLN)."""
    if graph_meta.get("projection") != "canonical_pathfinder_bounds":
        raise SourceSchemaError(
            f"unsupported graph projection: {graph_meta.get('projection')!r}"
        )
    min_x, min_z, max_x, max_z = _graph_bounds_xz(graph_meta, width=width, height=height)
    scale_x = (width - 1) / (max_x - min_x)
    return _projection_payload(
        min_x=min_x,
        min_z=min_z,
        max_x=max_x,
        max_z=max_z,
        width=width,
        height=height,
        scale_pixels_per_meter=float(graph_meta.get("scale_pixels_per_meter", scale_x)),
        graph_meta=graph_meta,
    )


def _build_projection(
    level_meta: dict[str, Any],
    graph_meta: dict[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    if level_meta.get("projection") != "canonical_pathfinder_bounds":
        raise SourceSchemaError(f"unsupported floorplan projection: {level_meta.get('projection')!r}")
    bounds = _mapping(level_meta.get("bounds"), "floorplan meta bounds")
    min_x = float(bounds["min_x"])
    min_z = float(bounds["min_z"])
    max_x = float(bounds["max_x"])
    max_z = float(bounds["max_z"])
    if max_x <= min_x or max_z <= min_z:
        raise SourceSchemaError(f"invalid floorplan bounds: {bounds}")
    if int(level_meta.get("width", 0)) != width or int(level_meta.get("height", 0)) != height:
        raise SourceSchemaError("floorplan meta dimensions do not match the six map images")

    graph_min_x, graph_min_z, graph_max_x, graph_max_z = _graph_bounds_xz(
        graph_meta, width=width, height=height
    )
    if not np.allclose(
        [min_x, min_z, max_x, max_z],
        [graph_min_x, graph_min_z, graph_max_x, graph_max_z],
        atol=1e-6,
        rtol=0.0,
    ):
        raise SourceSchemaError("floorplan and graph map bounds disagree")

    scale_x = (width - 1) / (max_x - min_x)
    return _projection_payload(
        min_x=min_x,
        min_z=min_z,
        max_x=max_x,
        max_z=max_z,
        width=width,
        height=height,
        scale_pixels_per_meter=float(level_meta.get("scale_pixels_per_meter", scale_x)),
        graph_meta=graph_meta,
    )


def _projection_payload(
    *,
    min_x: float,
    min_z: float,
    max_x: float,
    max_z: float,
    width: int,
    height: int,
    scale_pixels_per_meter: float,
    graph_meta: dict[str, Any],
) -> dict[str, Any]:
    scale_x = (width - 1) / (max_x - min_x)
    scale_z = (height - 1) / (max_z - min_z)
    graph_mpp = float(graph_meta.get("meters_per_pixel", 1.0 / scale_pixels_per_meter))
    return {
        "coordinate_frame": "habitat_world_xz",
        "pixel_order": ["u", "v"],
        "origin": "top_left",
        "u_axis": "+world_x",
        "v_axis": "+world_z",
        "rounding": "nearest_integer_then_clip",
        "width": width,
        "height": height,
        "bounds_xz": [min_x, min_z, max_x, max_z],
        "scale_pixels_per_meter": scale_pixels_per_meter,
        "meters_per_pixel": graph_mpp,
        "world_xz_to_pixel": [
            [scale_x, 0.0, -scale_x * min_x],
            [0.0, scale_z, -scale_z * min_z],
            [0.0, 0.0, 1.0],
        ],
    }


def _exact_overlay(split_root: Path, paths: list[str], pattern: str) -> Path:
    regex = re.compile(rf"^{pattern}$")
    matches = [value for value in paths if regex.fullmatch(Path(value).name)]
    if len(matches) != 1:
        raise SourceSchemaError(
            f"expected exactly one overlay matching {regex.pattern!r}, found {matches}"
        )
    return _source_path(split_root, matches[0])


def _source_path(split_root: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise SourceSchemaError(f"invalid source-relative path: {raw_path!r}")
    root = split_root.resolve()
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SourceSchemaError(f"source path escapes split root: {raw_path!r}") from exc
    return path


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SourceSchemaError(f"{label} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SourceSchemaError(f"missing JSON metadata: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SourceSchemaError(f"cannot read JSON metadata: {path}") from exc
    return _mapping(value, str(path))
