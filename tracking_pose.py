"""Measured pre-action tracking poses in Enactive's first-frame FLU coordinates."""

from typing import Any, Sequence

import numpy as np


def measured_local_poses(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, str]:
    """Return XYZ/xyzw poses; commands and terminal transitions are never integrated."""
    if not rows:
        raise ValueError("cannot convert an empty measured episode")
    positions, headings, sources = [], [], set()
    for index, row in enumerate(rows):
        try:
            teacher = row["teacher"]
            if "dt_scene" in teacher:
                pose = teacher["dt_scene"]["robot_pose"]
                position = np.asarray(pose["position_m"], dtype=np.float64)
                heading = float(pose["yaw_rad"])
                sources.add("teacher.dt_scene.robot_pose")
            else:
                position = np.asarray(teacher["robot_position"], dtype=np.float64)
                target = np.asarray(teacher["target_position"], dtype=np.float64)
                error = float(teacher["yaw_error_rad"])
                if target.shape != (3,) or not np.isfinite(target).all():
                    raise ValueError("invalid target position")
                if position.shape != (3,):
                    raise ValueError("invalid robot position")
                delta = target - position
                if np.linalg.norm(delta[[0, 2]]) < 1e-8:
                    raise ValueError("coincident target: STT yaw cannot be recovered")
                # Collector error is bearing-forward in world XZ (clockwise).
                # Enactive heading is atan2(-forward_z, forward_x), hence PLUS error.
                heading = np.arctan2(-delta[2], delta[0]) + error
                sources.add("teacher.robot_position+target_position+yaw_error_rad")
            if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(heading):
                raise ValueError("non-finite or invalid measured pose")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid measured pose at row {index}: {exc}") from exc
        positions.append(position)
        headings.append(heading)
    if len(sources) != 1:
        raise ValueError("measured pose source changes within episode")

    displacement = np.asarray(positions) - positions[0]
    yaw = np.unwrap(headings) - headings[0]
    c, s = np.cos(headings[0]), np.sin(headings[0])
    poses = np.zeros((len(rows), 7), dtype=np.float32)
    # Habitat is Y-up. First-frame body axes are forward, left, up.
    poses[:, 0] = c * displacement[:, 0] - s * displacement[:, 2]
    poses[:, 1] = -s * displacement[:, 0] - c * displacement[:, 2]
    poses[:, 2] = displacement[:, 1]
    poses[:, 5] = np.sin(yaw / 2)
    poses[:, 6] = np.cos(yaw / 2)
    return poses, sources.pop()
