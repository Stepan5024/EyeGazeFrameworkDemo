from dataclasses import dataclass
import datetime
from typing import List, Tuple
import numpy as np

@dataclass
class GazeResultContainer:
    def __init__(self,
                 pitch: np.ndarray,
                 yaw: np.ndarray,
                 bboxes: np.ndarray,
                 landmarks: np.ndarray,
                 scores: np.ndarray,
                 status: str,
                 image: np.ndarray,
                 coordinates: List[Tuple[float, float]],
                 datetime: datetime,
                 emotions: List[str]):
        self.pitch = pitch
        self.yaw = yaw
        self.bboxes = bboxes
        self.landmarks = landmarks
        self.scores = scores
        self.status = status
        self.image = image
        self.coordinates = coordinates
        self.datetime = datetime
        self.emotions = emotions