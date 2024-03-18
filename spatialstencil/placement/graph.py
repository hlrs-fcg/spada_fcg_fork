from typing import Sequence

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

from spatialstencil.placement.domain import FieldDomain
from spatialstencil.placement.partition import Placement
from spatialstencil.placement.stencil import Stencil, StencilDirection


class StencilGraph:

    DOMAIN = 'domain'
    STENCIL = 'stencil'
    FIELD_NAME = 'name'
    FIELD_VERSION = 'version'

    def __init__(self, graph: ig.Graph,
                 domain: FieldDomain,
                 field_domains: Sequence[FieldDomain],
                 field_names: Sequence[str],
                 field_versions: Sequence[int],
                 stencils: Sequence[Stencil]
                 ) -> None:

        self.graph = graph
        assert len(stencils) == len(graph.es)
        assert len(field_domains) == len(graph.vs)
        assert len(field_names) == len(graph.vs)
        assert len(field_versions) == len(graph.vs)
        self.graph.vs[StencilGraph.FIELD_NAME] = field_names
        self.graph.vs[StencilGraph.FIELD_VERSION] = field_versions
        self.graph.vs[StencilGraph.DOMAIN] = field_domains
        self.graph.es[StencilGraph.STENCIL] = stencils
        self.graph[StencilGraph.DOMAIN] = domain

    def domain(self) -> FieldDomain:
        return self.graph[StencilGraph.DOMAIN]

    def stencils(self) -> Sequence[Stencil]:
        return self.graph.es[StencilGraph.STENCIL]

    def edges(self) -> ig.EdgeSeq:
        return self.graph.es

    def out_edges(self, vertex) -> ig.EdgeSeq:
        return self.graph.es.select(_source=vertex)

    def in_edges(self, vertex) -> ig.EdgeSeq:
        return self.graph.es.select(_target=vertex)

    def plot(self, filename="stencil_graph.png"):
        # Plot the stencil graph
        layout = self.graph.layout_sugiyama()

        edge_label = [f"{s.shape}" for s in self.graph.es[StencilGraph.STENCIL]]
        vertex_color = "lightblue"
        vertex_label = [f"{name}#{version}" for (name, version) in zip(list(self.graph.vs[self.FIELD_NAME]),
                                                                       list(self.graph.vs[self.FIELD_VERSION]))]
        ig.plot(self.graph,
                layout=layout,
                vertex_label=vertex_label,
                vertex_size=80,
                vertex_color=vertex_color,
                edge_color=["#666" if d.direction == StencilDirection.PARALLEL
                            else "red" if d.direction == StencilDirection.FORWARD
                            else "blue" for d in self.graph.es[StencilGraph.STENCIL]],
                edge_width=1,
                edge_label=edge_label,
                bbox=(250 + len(self.graph.vs) * 60, 250 + len(self.graph.vs) * 60),
                margin=40,
                target=filename)


class PlacedStencilGraph(StencilGraph):

    def __init__(self,
                 stencil_graph: StencilGraph,
                 placement: Placement,
                 distances: Sequence[int]
                 ) -> None:
        super().__init__(stencil_graph.graph,
                         stencil_graph.domain(),
                         stencil_graph.graph.vs[StencilGraph.DOMAIN],
                         stencil_graph.graph.vs[StencilGraph.FIELD_NAME],
                         stencil_graph.graph.vs[StencilGraph.FIELD_VERSION],
                         stencil_graph.stencils())
        self.placement = placement
        self.distances = distances
        self.graph.vs['partition'] = self.placement.parts()

    def placement(self) -> Placement:
        return self.placement

    def plot(self, filename="stencil_graph.png"):
        # Plot the stencil graph
        layout = self.graph.layout_sugiyama()

        edge_label = [f"{d} ;\n {s.shape}" for (d, s) in zip(self.distances, self.graph.es[StencilGraph.STENCIL])]

        # Create a categorical color map for the partitions
        cmap = plt.get_cmap('tab20')
        vertex_color = [cmap(p + 1) for p in self.graph.vs['partition']]

        # Assign vertex labels based on the partition
        vertex_label = [f"{l}#{v}\n{p}" for (l, v, p) in zip(self.graph.vs[self.FIELD_NAME],
                                                             self.graph.vs[self.FIELD_VERSION],
                                                             self.placement.offsets)]

        ig.plot(self.graph,
                layout=layout,
                vertex_label=vertex_label,
                vertex_size=100,
                vertex_color=vertex_color,
                edge_color=["#666" if d.direction == StencilDirection.PARALLEL
                            else "red" if d.direction == StencilDirection.FORWARD
                            else "blue" for d in self.graph.es[StencilGraph.STENCIL]],
                edge_width=1,
                edge_label=edge_label,
                bbox=(250 + len(self.graph.vs) * 60, 250 + len(self.graph.vs) * 60),
                margin=60,
                target=filename)