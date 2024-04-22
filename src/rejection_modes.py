from enum import Enum


class RejectionModes(Enum):
    """The available rejection modes for the buffer"""

    FULL = 1
    PARTIAL = 2