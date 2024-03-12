from typing import Sequence
import igraph as ig
import numpy as np

from spatialstencil.placement.domain import FieldDomain
from spatialstencil.placement.stencil import Stencil, StencilDirection


class StencilGraph:

    DOMAIN = 'domain'
    STENCIL = 'stencil'
    FIELD_NAME = 'name'

    def __init__(self, graph: ig.Graph,
                 domain: FieldDomain,
                 field_domains: Sequence[FieldDomain],
                 stencils: Sequence[Stencil]
                 ) -> None:

        self.graph = graph
        for v in self.graph.vs:
            assert v[StencilGraph.FIELD_NAME] is not None
        assert len(stencils) == len(graph.es)
        assert len(field_domains) == len(graph.vs)
        self.graph.vs[StencilGraph.DOMAIN] = field_domains
        self.graph.es[StencilGraph.STENCIL] = stencils
        self.graph[StencilGraph.DOMAIN] = domain
        # Check that for all forward edges the xy directions are 0 and the z direction is negative
        for i, edge in enumerate(self.graph.es):
            if edge[StencilGraph.STENCIL].direction == StencilDirection.FORWARD:
                assert np.all(edge[StencilGraph.STENCIL].shape[:, 0:2] == 0)
                assert np.all(edge[StencilGraph.STENCIL].shape[:, 2] < 0)
        # Check for all backward edges the xy directions are 0 and the z direction is positive
        for i, edge in enumerate(self.graph.es):
            if edge[StencilGraph.STENCIL].direction == StencilDirection.BACKWARD:
                assert np.all(edge[StencilGraph.STENCIL].shape[:, 0:2] == 0)
                assert np.all(edge[StencilGraph.STENCIL].shape[:, 2] > 0)

    def domain(self) -> FieldDomain:
        return self.graph[StencilGraph.DOMAIN]

    def stencils(self) -> Sequence[Stencil]:
        return self.graph.es[StencilGraph.STENCIL]

    def edges(self):
        return self.graph.es

    def out_edges(self, vertex):
        return self.graph.es.select(_source=vertex)

    def in_edges(self, vertex):
        return self.graph.es.select(_target=vertex)

    def plot(self, filename="stencil_graph.png"):
        # Plot the stencil graph
        layout = self.graph.layout_reingold_tilford(mode="in")
        ig.plot(self.graph,
                layout=layout,
                vertex_label=self.graph.vs["name"],
                vertex_size=30,
                vertex_color="lightblue",
                edge_color=["black" if d.direction == StencilDirection.PARALLEL else "red" if d.direction == StencilDirection.FORWARD else "blue" for d in self.graph.es[StencilGraph.STENCIL]],
                edge_width=1,
                edge_label=[f"{s.shape}" for s in self.graph.es[StencilGraph.STENCIL]],
                bbox=(500, 500),
                margin=20,
                target=filename)