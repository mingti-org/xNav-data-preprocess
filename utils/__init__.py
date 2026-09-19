from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, Tuple

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

def get_task_idx(ds: LeRobotDataset, task: str) -> int:
    """Get the index of a task, adding it if it doesn't exist."""
    task_index = ds.meta.get_task_index(task)
    if task_index is None:
        ds.meta.add_task(task)
        task_index = ds.meta.get_task_index(task)
    return task_index

class Traj:
    """
    Single trajectory, iterable to output each frame.
    """
    def __init__(self, frames):
        raise NotImplementedError("Traj is an abstract class and cannot be instantiated directly.")

    def __len__(self)-> int:
        raise NotImplementedError("Traj is an abstract class and cannot be instantiated directly.")

    def __iter__(self) -> Iterable[Tuple[dict, str]]:
        raise NotImplementedError("Traj is an abstract class and cannot be instantiated directly.")
    
    @property
    def metadata(self) -> dict:
        """Return metadata for the trajectory."""
        raise NotImplementedError("Traj is an abstract class and cannot be instantiated directly.")

class Trajectories(ABC):
    """
    abstract class representing a collection of trajectories.
    """
    FPS: int = None
    ROBOT_TYPE: str = None
    FEATURES: dict = None
    INSTRUCTION_KEY: str = None
    
    def __init__(self, data_path: str):
        raise NotImplementedError("Trajectories is an abstract class and cannot be instantiated directly.")

    def __len__(self)-> int:
        raise NotImplementedError("Trajectories is an abstract class and cannot be instantiated directly.")

    def __iter__(self)-> Iterable[Traj]:
        raise NotImplementedError("Trajectories is an abstract class and cannot be instantiated directly.")
    
    @property
    @abstractmethod
    def schema(self) -> dict:
        """Return the schema for the dataset."""
        raise NotImplementedError("Trajectories is an abstract class and cannot be instantiated directly.")
