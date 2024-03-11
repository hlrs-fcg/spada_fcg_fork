"""
Defines placements and the cost model of a placement.
"""
from dataclasses import dataclass

import numpy as np

from spatialstencil.placement.graph import StencilDirection, StencilGraph


@dataclass
class Placement:
    """
    A placement is a mapping from fields / vertices of a StencilGraph
    to offsets and strides in the domain of the device.
    each offset and stride is represented as 1x2 arrays of integers.

    To place a stencil graph on our 2-dimensional PE grid, each field $v$ is associated with:
    \begin{itemize}
        \item An offset $O(v)=(O_x(v), O_y(v))$
        \item A stride $I(v)=(I_x(v), I_y(v))$.
    \end{itemize}
    We place each cell of a field $v$ onto a PE by placing the $(j, k)$-th column
    of $v$ at position $O(v) + (j \cdot I_x(v) , k \cdot  I_y(v) )$.

    # column 0 -> x, column 1 -> y
    """
    offsets: np.ndarray
    strides: np.ndarray

    def __post_init__(self):
        assert self.offsets.shape[1] == 2
        assert self.strides.shape[1] == 2
        assert self.offsets.shape[0] == self.strides.shape[0]
        assert np.issubdtype(self.offsets.dtype, np.integer)
        assert np.issubdtype(self.strides.dtype, np.integer)

    def __eq__(self, other):
        return np.array_equal(self.offsets, other.offsets) and np.array_equal(self.strides, other.strides)


class CostModel:

    def __init__(self, stencil_graph):
        self.stencil_graph = stencil_graph

    def cost(self, placement: Placement) -> float:
        """
        Calculate the cost of a placement
        :param placement: Placement
        :return: float
        """
        raise NotImplementedError

    def edge_distance_of_placement(self, placement: Placement) -> np.ndarray:
        """
        Calculate the distance of each edge for a given placement.
        :param placement: Placement
        :return: for each edge, its distance under the placement
        """
        # Iterate over all edges in order
        # For each edge, calculate the distance
        # Return the distances as a numpy array
        distances = np.zeros(len(self.stencil_graph.edges()), dtype=np.float32)
        for e in self.stencil_graph.edges():
            delta = self.distance_vector_of_edge(placement, e)
            # Calculate the l1 norm of each row
            # then take the maximum across the columns
            # to get the maximum distance of the edge
            distances[e.index] = np.max(delta)

        return distances

    @staticmethod
    def distance_vector_of_edge(placement: Placement, e) -> np.ndarray:
        """
        Calculate the distance vector of an edge for a given placement.
        This is the distance of each element of the stencil shape of the edge.

        For the case where the strides of two fields $u$ and $v$ are the same,
        we can compute the distance of a dependency edge $(u, v)$ as follows:
        Consider the stencil shape
        $S(e)=\{(\delta_x^{(0)}, \delta_y^{(0)}, \delta_z^{(0)}), \dotsc\}$ and let $\delta^{(i)} =(\delta_x^{(0)}, \delta_y^{(0)})$.
        The distance vector $\Delta(e)^{(i)}$ is given by:
        $$\Delta(e)^{(i)} = O(v)-O(u) - I(u) \odot \delta^{(i)} \enspace .$$
        Note that this formulation works for both horizontal and vertical stencils and that $\delta_z$ does not influence the computation.

        :param placement:
        :param e: edge in the graph
        :return: An array of distances, of size equal to the number of elements in the stencil
        """

        # Calculate the distance of the edge
        # and store it in the result array
        head = e.source
        tail = e.target

        # Get the offsets and strides of the fields
        offset_head = placement.offsets[head]
        offset_tail = placement.offsets[tail]
        stride_head = placement.strides[head]

        # Assert strides of head and tail match
        assert np.array_equal(placement.strides[tail], stride_head)

        # Get the stencil shape [(delta_x, delta_y)] of the edge
        delta_xy = e[StencilGraph.STENCIL].shape[:, :2]

        # Calculate the distance vector
        # Note that the offset and stride are 1x2 arrays
        # and the delta is a kx2 array
        # The result is a kx2 array
        # this works by broadcasting the 1x2 arrays to kx2 arrays
        # and then element-wise multiplication
        delta = offset_tail - offset_head - stride_head * delta_xy

        # Now, we compute the Manhattan distance across axis 1
        # To get a 1D array
        distances = np.sum(np.abs(delta), axis=1)
        return distances

    def distance_of_placement(self, distances):
        """
        Calculate the maximum distance of a placement (longest path in the graph, weighted by distances)

        The longest path in the graph weighted by distance is the maximum distance of the placed stencil program.
        :param distances:
        :return:
        """
        # uses the linear-time dynamic programming approach to finding the longest path in a DAG
        # assert edge weights have correct size
        assert len(distances) == len(self.stencil_graph.edges())
        # get the topological order of the graph
        #  If "in", all vertices come before their ancestors.
        top_order = self.stencil_graph.graph.topological_sorting(mode="IN")

        # initialize the distance array
        max_distance = np.zeros(len(self.stencil_graph.graph.vs), dtype=np.float32)

        for v in top_order:
            # get the maximum distance of the incoming edges
            # and add the distance of the current edge
            for e in self.stencil_graph.graph.vs[v].out_edges():
                max_distance[v] = max(max_distance[v], max_distance[e.target] + distances[e.index])

        return np.max(max_distance)

    def energy_of_placement(self, placement: Placement) -> float:
        """
        For the case where the strides of every field connected by a dependency edge is the same, we use the same
        distance vector computation and sum all the contributions. Each distance vector is multiplied by the
        communication volume. This equals the size of the domain in the case of horizontal stencils and a single x-y
        plane in the case of vertical stencils.
        Then, we sum all edge costs to get the total energy.

        :param placement:
        :return:
        """
        energy = np.zeros_like(self.stencil_graph.edges())
        for e in self.stencil_graph.edges():
            distances = self.distance_vector_of_edge(placement, e)

            # Energy is distance times volume,
            # summed over each stencil element
            energy[e.index] = np.sum(distances) * self.communication_volume_of_edge(e)

        return energy.sum()

    def local_communication_volume_of_edge(self, e):
        """
        The local communication volume is the number of elements communicated by each edge
        # times the height of the z column in the case of horizontal stencils
        :param e: 
        :return: 
        """
        communication_volume = e['stencil'].shape.shape[0]
        domain = self.stencil_graph.graph.vs[e.source]['domain']
        direction = e['direction']
        if direction == StencilDirection.PARALLEL:
            communication_volume *= domain.z_length()

        return communication_volume

    def communication_volume_of_edge(self, e):
        # the number of elements communicated by each edge is the number of elements in the stencil shape
        # times the volume of the domain in the case of horizontal stencils
        # and a single x-y plane in the case of vertical stencils
        communication_volume = e[StencilGraph.STENCIL].shape.shape[0]

        domain = self.stencil_graph.graph.vs[e.source][StencilGraph.DOMAIN]
        direction = e[StencilGraph.STENCIL_DIRECTION]
        if direction == StencilDirection.PARALLEL:
            communication_volume *= domain.volume()
        else:
            communication_volume *= domain.xy_plane_area()

        return communication_volume

    def contention_of_placement(self, placement: Placement) -> float:
        """
        The input contention of a field is the sum of the communication volumes of its incoming edges.
        Similarly, the output contention of a field is the sum of the communication volume of its outgoing edges.

        We assume that each pair of fields is either mapped to a disjoint or the same set of PEs. Then, to compute
        the most congested PE we need to group the fields according to if they map to the same PEs. This can be done
        by  hashing the fields into buckets based on their top-left coordinate. Then, the input contention of the
        placed stencil program is the largest sum of input contentions of the fields in any given bucket. Proceed
        similarly for the output contention. The contention is the maximum of input and output contention.

        :param placement: Placement
        :return: float The contention of the placement
        """

        equivalence_classes = dict()

        for v in self.stencil_graph.graph.vs:
            offset = placement.offsets[v.index]
            tuple = (offset[0], offset[1])
            if tuple not in equivalence_classes:
                equivalence_classes[tuple] = []
            equivalence_classes[tuple].append(v)

        input_contention = 0
        output_contention = 0
        for key in equivalence_classes:
            input_contention = max(input_contention, self.input_contention_of_fields(equivalence_classes[key]))
            output_contention = max(output_contention, self.output_contention_of_fields(equivalence_classes[key]))

        return max(input_contention, output_contention)

    def input_contention_of_fields(self, fields):
        """
        The input contention of a field is the sum of the communication volumes of its incoming edges.
        :param fields: list of fields
        :return: float
        """
        contention = 0.0
        for v in fields:
            for e in self.stencil_graph.in_edges(v):
                assert e.target == v.index
                contention += self.local_communication_volume_of_edge(e)
        return contention

    def output_contention_of_fields(self, fields):
        """
        The output contention of a field is the sum of the communication volume of its outgoing edges.
        :param fields: list of fields
        :return: float
        """
        contention = 0.0
        for v in fields:
            for e in self.stencil_graph.out_edges(v):
                assert e.source == v.index
                contention += self.local_communication_volume_of_edge(e)
        return contention

    def depth_of_placement(self, placement: Placement) -> float:
        """
        Calculate the depth of a placement
        :param placement: Placement
        :return: float
        """
        raise NotImplementedError
