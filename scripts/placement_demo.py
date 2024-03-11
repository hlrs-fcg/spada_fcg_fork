import numpy as np
import igraph as ig

from spatialstencil.placement.graph import StencilShape, StencilDirection, FieldDomain, StencilGraph
from spatialstencil.placement.model import CostModel
from spatialstencil.placement.partition import FieldPartition


def demo_graph():

    domain = np.array([[0, 0, 0], [256, 256, 64]], dtype=np.int32)
    print(domain)
    domain_type = FieldDomain(domain)

    five_point_stencil = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.int32)

    # one_point_stencil_right = np.array( [[1, 0, 0]], dtype=np.int32)
    one_point_stencil_up = np.array([[0, 1, 0]], dtype=np.int32)

    z_stencil_forward = np.array([[0, 0, -1]], dtype=np.int32)
    z_stencil_backward = np.array([[0, 0, 1]], dtype=np.int32)

    # Create StencilGraph with 8 nodes
    # and an inverted tree like shape
    # u -> w, v-> w, x -> z, y -> z, w -> a, z -> a, a -> b, b -> c
    # as numbers, we get
    # u = 0, v = 1, w = 2, x = 3, y = 4, z = 5, a = 6, b= 7, c = 8
    g = ig.Graph(directed=True, n=9)
    g.add_edges([(0, 2), (1, 2), (3, 5), (4, 5), (2, 6), (5, 6), (6, 7), (7, 8), (0, 8)])
    # Set the names of the nodes
    g.vs["name"] = ["u", "v", "w", "x", "y", "z", "a", "b", "c"]

    # We alternate between horizontal (five point) and vertical stencils
    # so the stencils w-> a and z-> a and b-> c are vertical
    # and the rest are horizontal
    stencils = [StencilShape(five_point_stencil),
                StencilShape(five_point_stencil),
                StencilShape(five_point_stencil),
                StencilShape(five_point_stencil),
                StencilShape(z_stencil_forward),
                StencilShape(z_stencil_forward),
                StencilShape(five_point_stencil),
                StencilShape(z_stencil_backward),
                StencilShape(one_point_stencil_up)]

    # All horizontal stencils are parallel, while the first vertical stencils are forward and the last is backward
    stencil_directions = [StencilDirection.PARALLEL,
                          StencilDirection.PARALLEL,
                          StencilDirection.PARALLEL,
                          StencilDirection.PARALLEL,
                          StencilDirection.FORWARD,
                          StencilDirection.FORWARD,
                          StencilDirection.PARALLEL,
                          StencilDirection.BACKWARD,
                          StencilDirection.PARALLEL]

    # Create the StencilGraph
    stencil_graph = StencilGraph(g, domain_type, [domain_type] * 9, stencils, stencil_directions)

    return stencil_graph


def demo_placement_interleave():
    # Create a placement with two partitions, one with u, v, w, a, b, c and the other with x, y, z
    # The partitions are placed on the same processing elements
    # as numbers, we get
    # u = 0, v = 1, w = 2, x = 3, y = 4, z = 5, a = 6, b= 7, c = 8
    # we use offset (0, 0) for the first partition
    # and offset (0, 1) for the second partition
    # the stride is (2, 1) for both partitions
    parts = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    partition = FieldPartition(parts)
    placement = partition.place_interleaved()
    return placement


def demo_placement_separated(domain: FieldDomain):
    """
    Places the nodes in the graph on two partitions of separate processing elements
    :param domain:
    :return:
    """
    parts = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    # manual result
    offsets = np.array([[0, 0], [0, 0], [0, 0], [domain.x()[1], 0], [domain.x()[1], 0], [domain.x()[1], 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    strides = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]], dtype=np.int32)
    partition = FieldPartition(parts)
    # automatic result
    placement = partition.place_blocked(domain)
    # check if the automatic result is the same as the manual result
    assert np.allclose(placement.offsets, offsets)
    assert np.allclose(placement.strides, strides)
    return placement


def demo_costs(stencil_graph, place):
    print(place)
    cost_model = CostModel(stencil_graph)
    distances = cost_model.edge_distance_of_placement(place)
    print(f"distances {distances}")
    max_distances = cost_model.distance_of_placement(distances)
    print(f"max distances {max_distances}")
    stencil_graph.graph.es['max_distance'] = max_distances

    energy = cost_model.energy_of_placement(place)
    print(f"energy: {energy}")

    contention = cost_model.contention_of_placement(place)
    print(f"contention: {contention} \n")


def main():

    stencil_graph = demo_graph()
    stencil_graph.plot()

    print("Interleaved placement")
    place = demo_placement_interleave()
    demo_costs(stencil_graph, place)

    print("Separated placement")
    place2 = demo_placement_separated(stencil_graph.graph['domain'])
    demo_costs(stencil_graph, place2)


if __name__ == "__main__":
    main()
