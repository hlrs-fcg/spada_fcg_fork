"""
Represents a partition of the set of fields into disjoint subsets of fields.
These fields are then placed on the same processing element.
"""
from dataclasses import dataclass
from spatialstencil.placement.model import Placement
import numpy as np


@dataclass
class FieldPartition:
    """
    Represents a partition of the set of fields into disjoint subsets of fields with a spatial preference.
    These fields are meant to placed on the same sets of processing elements.
    """

    # n x 2 array
    # where there is a row for each field
    # and two integers denoting the (x, y) position of the partition it belongs to
    # the number of partitions is implicit in the number of distinct (x, y) values
    # column 0 -> x, column 1 -> y
    part: np.ndarray

    def __post_init__(self):
        # assert integer type
        assert np.issubdtype(self.part.dtype, np.integer)
        # assert shape
        assert self.part.shape[1] == 2
        assert self.part.shape[0] > 0

    def place_interleaved(self) -> Placement:
        """
        A vertex at position (x, y) in the partition gets the offset (x, y) in the placement
        the strides are given by creating a rectangle around the (x, y) coordinates
        :return:
        """
        def stride(a: np.ndarray) -> np.ndarray:
            return np.abs(np.max(a) - np.min(a))

        stride_x = stride(self.part[:, 0])
        stride_y = stride(self.part[:, 1])

        offsets = self.part.copy()
        strides = np.zeros_like(offsets)
        strides[:, 0] = stride_x + 1
        strides[:, 1] = stride_y + 1
        return Placement(offsets=offsets, strides=strides)
