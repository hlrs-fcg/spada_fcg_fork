from typing import Tuple

from spatialstencil.placement.graph import StencilGraph, PlacedStencilGraph
from spatialstencil.placement.model import PlacementCost, CostModel
from spatialstencil.placement.partition import FieldPartition


def best_of_k_placement(g: StencilGraph, k: int, shape: Tuple[int, int]) -> Tuple[PlacedStencilGraph, PlacementCost]:
    """
    Find the best placement of the graph g by trying k different placements and returning the best one.
    :param g:
    :param k:
    :param shape:
    :return:
    """
    assert k > 0
    best_cost = float('inf')
    best_cost_obj = None
    best_placement = None
    best_distances = None
    cost_model = CostModel(g)
    for i in range(k):
        partition_auto = FieldPartition.from_mla(g.graph, shape)
        place_auto = partition_auto.place_interleaved()
        cost = cost_model.cost(place_auto)
        distances = cost_model.edge_distance_of_placement(place_auto)
        print(cost)
        if cost.overall < best_cost or cost.overall == best_cost and cost.energy_over_links < best_cost_obj.energy_over_links:
            best_cost = cost.overall
            best_placement = place_auto
            best_cost_obj = cost
            best_distances = distances
        placed_graph = PlacedStencilGraph(g, place_auto, distances)
        # For debugging, plot the placement
        placed_graph.plot(g.graph['name'] + "_auto_" + str(i) + ".png")

    placed_graph = PlacedStencilGraph(g, best_placement, best_distances)
    return placed_graph, best_cost_obj
