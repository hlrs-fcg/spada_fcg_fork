import numpy as np
import igraph as ig
from spatialstencil.placement.domain import FieldDomain
from spatialstencil.placement.stencil import Stencil, StencilDirection
from spatialstencil.placement.graph import StencilGraph

def horizontal_diffusion():
    """
    Simplified example of a 2D diffusion stencil corresponding to the following GT4Py code:
    There is a vertex for each input field, output field and coefficient field. The edges represent the data dependencies.
    there is an edge for every field being used on the right hand side of an assignment

    def horizontal_diffusion(in_field: Field3D, out_field: Field3D, coeff: Field3D):
        with computation(PARALLEL), interval(...):

            lap_field = 4.0 * in_field[0, 0, 0] - (
                in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0]
            )
            res_x = lap_field[1, 0, 0] - lap_field[0, 0, 0]
            cond_x = res_x[0, 0, 0] * (in_field[1, 0, 0] - in_field[0, 0, 0])
            flx_field = 0 if cond_x > 0 else res_x

            res_y = lap_field[0, 1, 0] - lap_field[0, 0, 0]
            cond_y = res_y[0, 0, 0] * (in_field[0, 1, 0] - in_field[0, 0, 0])
            fly_field = 0 if cond_y > 0 else res_y

            out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
                flx_field[0, 0, 0] - flx_field[-1, 0, 0] + fly_field[0, 0, 0] - fly_field[0, -1, 0]
            )
    """

    # Define the domain
    domain = FieldDomain(np.array([[0, 0, 0], [256, 256, 50]], dtype=np.int32))

    # Define the stencils
    stencils = [
        # in_field -> lap_field
        Stencil(np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # lap_field -> res_x
        Stencil(np.array([[1, 0, 0], [0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # res_x -> cond_x
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # in_field -> cond_x
        Stencil(np.array([[1, 0, 0], [0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # res_x -> flx_field
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # cond_x -> flx_field
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),

        # lap_field -> res_y
        Stencil(np.array([[0, 1, 0], [0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # res_y -> cond_y
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # in_field -> cond_y
        Stencil(np.array([[0, 1, 0], [0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # res_y -> fly_field
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # cond_y -> fly_field
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),

        # in_field -> out_field
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # coeff -> out_field
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # flx_field -> out_field
        Stencil(np.array([[0, 0, 0], [-1, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # fly_field -> out_field
        Stencil(np.array([[0, 0, 0], [0, -1, 0]], dtype=np.int32), StencilDirection.PARALLEL),
    ]
    # 11 vertices in_field = 0, out_field = 1, coeff = 2, lap_field = 3, res_x = 4, cond_x = 5, flx_field = 6,
    # res_y = 7, cond_y = 8, fly_field = 9
    g = ig.Graph(directed=True, n=10, edges=[
        (0, 3), (3, 4), (4, 5), (0, 5), (4, 6), (5, 6), (3, 7), (7, 8), (0, 8), (7, 9), (8, 9), (0, 1), (2, 1), (6, 1), (9, 1)
    ])
    # Set field names
    g.vs["name"] = ["in_field", "out_field", "coeff", "lap_field", "res_x", "cond_x", "flx_field", "res_y", "cond_y", "fly_field"]

    # Define the graph
    graph = StencilGraph(g, domain, [domain] * 10, stencils)

    return graph


