"""
Represents a partition of the set of fields into disjoint subsets of fields.
These fields are then placed on the same processing element.
"""
from dataclasses import dataclass

from numpy.typing import NDArray

from spatialstencil.placement.graph import FieldDomain
import numpy as np


@dataclass
class Placement:
    """
    A placement is a mapping from fields / vertices of a StencilGraph
    to offsets and strides in the domain of the device.
    each offset and stride is represented as 1x2 arrays of integers.

    To place a stencil graph on our 2-dimensional PE grid, each field $v$ is associated with:
        * An offset O(v)=(O_x(v), O_y(v))
        * A stride I(v)=(I_x(v), I_y(v))
    We place each cell of a field $v$ onto a PE by placing the $(j, k)$-th column
    of v at position O(v) + (j * I_x(v) , k * I_y(v) ).

    # column 0 -> x, column 1 -> y
    """
    offsets: NDArray[np.int32]
    strides: NDArray[np.int32]

    def __post_init__(self):
        assert self.offsets.shape[1] == 2
        assert self.strides.shape[1] == 2
        assert self.offsets.shape[0] == self.strides.shape[0]
        assert np.issubdtype(self.offsets.dtype, np.integer)
        assert np.issubdtype(self.strides.dtype, np.integer)

    def __eq__(self, other):
        return np.array_equal(self.offsets, other.offsets) and np.array_equal(self.strides, other.strides)

    def unique_offsets(self) -> NDArray[np.int32]:
        return np.unique(self.offsets, axis=0)

    def edge_crosses_partition(self, e) -> float:
        """
        Calculate the weight of an edge in the depth calculation.
        An edge that crosses a partition has weight 1, an edge that does not cross a partition has weight 0.
        :param e: edge in the graph
        :param placement: Placement
        :return: float   1 if the edge crosses a partition, 0 otherwise
        """
        head = e.source
        tail = e.target
        offset_head = self.offsets[head]
        offset_tail = self.offsets[tail]
        return 1.0 if not np.array_equal(offset_head, offset_tail) else 0.0


@dataclass
class FieldPartition:
    """
    Represents a partition of the set of fields into disjoint subsets of fields with a spatial preference.
    These fields are meant to placed on the same sets of processing elements.
    """

    # n == number of fields
    # n x 2 array
    # where there is a row for each field
    # and two integers denoting the (x, y) position of the partition it belongs to
    # the number of partitions is implicit in the number of distinct (x, y) values
    # column 0 -> x, column 1 -> y
    part: NDArray[np.int32]

    def __post_init__(self):
        # assert integer type
        assert np.issubdtype(self.part.dtype, np.integer)
        # assert shape
        assert self.part.shape[1] == 2
        assert self.part.shape[0] > 0

    def place_interleaved(self) -> Placement:
        """
        In the interleaving placement strategy, the partitions of fields are placed at small offsets to each other in
        an interleaved fashion. The advantage of this approach is that none of the distances are very large. However,
        the average distance is increased for all stencils, even those horizontal stencils that do not cross the
        partition.

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

    def place_blocked(self, domain: FieldDomain) -> Placement:
        """
        In the blocked placement, each partition is assigned to a disjoint area of the chip with stride 1.
        This means that all stencils that do not cross a partition can be executed optimally.
        However, stencils that cross the partition may travel a large distance proportional to the domain side length.

        For example, consider 2 partitions, 2x4 8 PEs and a 2D domain of size 2x2
        Then, we will have the following placement
         0 0 1 1
         0 0 1 1
        For 4x2 PEs, we get the following placement
         0 0
         0 0
         1 1
         1 1
        :return:
        """
        # TODO Assumes all domains are the same
        # If domains are not the same, we might need to broadcast and this affects placement...
        # What if it's a single column? -> then we broadcast it so it's 3D again.
        x_offset_multiplier = domain.x_length()
        y_offset_multiplier = domain.y_length()
        strides = np.ones_like(self.part)
        offsets = self.part.copy() * np.array([x_offset_multiplier, y_offset_multiplier])
        return Placement(offsets=offsets, strides=strides)
