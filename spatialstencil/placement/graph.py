from dataclasses import dataclass
from enum import Enum
from typing import Sequence
import igraph as ig
import numpy as np


@dataclass
class FieldDomain:
    """
    Represents the 3D domain of a field

    Represented as a 2 by 3 array
    column 0 -> x, column 1 -> y, column 2 -> z
    The first row is the lower bound of the domain
    The second row is the upper bound of the domain
    """
    domain: np.ndarray

    def x(self):
        """
        Returns the x coordinate upper and lower bounds of the domain
        :return:
        """
        return self.domain[:, 0]

    def y(self):
        """
        Returns the y coordinate upper and lower bounds of the domain
        :return:
        """
        return self.domain[:, 1]

    def z(self):
        """
        Returns the z coordinate upper and lower bounds of the domain
        :return:
        """
        return self.domain[:, 2]

    # Check if a domain is valid
    # A domain is valid if it is a 2x3 shaped array and all its elements are integers
    # column 0 -> x, column 1 -> y, column 2 -> z
    def __post_init__(self):
        assert self.domain.shape == (2, 3)
        assert np.issubdtype(self.domain.dtype, np.integer)
        # Check that the lower bound is less than or equal to the upper bound
        assert np.all(self.domain[0] <= self.domain[1])

    def __eq__(self, other):
        return np.array_equal(self.domain, other.domain)

    def x_length(self):
        return self.domain[1][0] - self.domain[0][0]

    def y_length(self):
        return self.domain[1][1] - self.domain[0][1]

    def z_length(self):
        return self.domain[1][2] - self.domain[0][2]

    def xy_plane_area(self):
        return self.x_length() * self.y_length()

    def volume(self):
        return self.z_length() * self.xy_plane_area()



@dataclass
class StencilShape:
    # Represented as a k by 3 array
    # column 0 -> x, column 1 -> y, column 2 -> z
    shape: np.ndarray

    # Check if a stencil is valid
    # A stencil is valid if it is a k times 3 shaped array and all its elements are integers
    def __post_init__(self):
        assert self.shape.shape[1] == 3
        assert self.shape.shape[0] > 0
        assert np.issubdtype(self.shape.dtype, np.integer)

    def __eq__(self, other):
        return np.array_equal(self.shape, other.shape)

    def is_horizontal(self) -> bool:
        """
        A stencil is horizontal if all z values (3rd coordinate) are 0
        :return: bool   True if the stencil is horizontal, False otherwise
        """
        return np.all(self.shape[:, 2] == 0)

    def is_vertical(self) -> bool:
        """
        A stencil is vertical if all x and y values (1st and 2nd coordinates) are 0
        :return: bool   True if the stencil is vertical, False otherwise
        """
        return np.all(self.shape[:, 0:2] == 0)


class StencilDirection(Enum):
    """
    Indicates the order in which the z-coordinate of the stencil is traversed
    """
    # The stencil is traversed in parallel
    PARALLEL = 0
    # The stencil is traversed in a forward direction (increasing z)
    FORWARD = 1
    # The stencil is traversed in a backward direction (decreasing z)
    BACKWARD = 2


class StencilGraph:

    DOMAIN = 'domain'
    STENCIL = 'stencil'
    STENCIL_DIRECTION = 'direction'
    FIELD_NAME = 'name'

    def __init__(self, graph: ig.Graph,
                 domain: FieldDomain,
                 field_domains: Sequence[FieldDomain],
                 stencils: Sequence[StencilShape],
                 stencil_directions: Sequence[StencilDirection]
                 ) -> None:

        self.graph = graph
        for v in self.graph.vs:
            assert v[StencilGraph.FIELD_NAME] is not None
        assert len(stencils) == len(graph.es)
        assert len(field_domains) == len(graph.vs)
        self.graph.vs[StencilGraph.DOMAIN] = field_domains
        self.graph.es[StencilGraph.STENCIL] = stencils
        self.graph[StencilGraph.DOMAIN] = domain
        assert len(stencil_directions) == len(graph.es)
        self.graph.es[StencilGraph.STENCIL_DIRECTION] = stencil_directions
        # Check that for all forward edges the xy directions are 0 and the z direction is negative
        for i, edge in enumerate(self.graph.es):
            if edge[StencilGraph.STENCIL_DIRECTION] == StencilDirection.FORWARD:
                assert np.all(edge[StencilGraph.STENCIL].shape[:, 0:2] == 0)
                assert np.all(edge[StencilGraph.STENCIL].shape[:, 2] < 0)
        # Check for all backward edges the xy directions are 0 and the z direction is positive
        for i, edge in enumerate(self.graph.es):
            if edge[StencilGraph.STENCIL_DIRECTION] == StencilDirection.BACKWARD:
                assert np.all(edge[StencilGraph.STENCIL].shape[:, 0:2] == 0)
                assert np.all(edge[StencilGraph.STENCIL].shape[:, 2] > 0)

    def edges(self):
        return self.graph.es

    def out_edges(self, vertex):
        return self.graph.es.select(_source=vertex)

    def in_edges(self, vertex):
        return self.graph.es.select(_target=vertex)

    def plot(self):
        # Plot the stencil graph
        layout = self.graph.layout_reingold_tilford(mode="in")
        ig.plot(self.graph,
                layout=layout,
                vertex_label=self.graph.vs["name"],
                vertex_size=30,
                vertex_color="lightblue",
                edge_color=["black" if d == StencilDirection.PARALLEL else "red" if d == StencilDirection.FORWARD else "blue" for d in self.graph.es[StencilGraph.STENCIL_DIRECTION]],
                edge_width=1,
                edge_label=[f"{s.shape}" for s in self.graph.es[StencilGraph.STENCIL]],
                bbox=(500, 500),
                margin=20,
                target="stencil_graph.png")