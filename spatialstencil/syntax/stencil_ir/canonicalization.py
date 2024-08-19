"""
Includes canonicalization routines for Stencil IR.
"""

from collections import defaultdict
from spatialstencil.syntax.stencil_ir import irnodes as sast
from spatialstencil.syntax.stencil_ir import type_inference


def canonicalize(program: sast.Program) -> sast.Program:
    """
    Canonicalizes a Stencil IR program by reordering extents to a canonical order.
    """
    program = CanonicalizeExtents().visit(program)
    return program


class CanonicalizeExtents(sast.NodeTransformer):
    """
    Sorts extents to canonicalize order.
    """

    def _extent_set_from_node(
        self,
        extents: list[sast.OffsetAndInterval],
    ) -> dict[tuple[int | None], set[tuple[int | None]]]:
        # Collect all extents as a dictionary of interval -> set-of-values
        result: dict[tuple[int | None], set[tuple[int | None]]] = defaultdict(set)
        for extent in extents:
            result[extent.interval].add(extent.values)
        return result

    def _modify_typeinfo(self, operation_type: sast.OperationType):
        """
        Helper function that updates the type information based on inferred types.
        """
        for src in operation_type.source:
            if isinstance(src, sast.FieldType):
                extent_set = self._extent_set_from_node(src.extent.extents)
                src.extent.extents = [
                    sast.OffsetAndInterval(ex, interval) for interval, ex in type_inference.sort_extents(extent_set)
                ]

        if operation_type.destination:
            for dst in operation_type.destination:
                extent_set = self._extent_set_from_node(dst.extent.extents)
                if isinstance(dst, sast.FieldType):
                    dst.extent.extents = [
                        sast.OffsetAndInterval(ex, interval) for interval, ex in type_inference.sort_extents(extent_set)
                    ]

    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        self._modify_typeinfo(node.operation_type)
        return self.generic_visit(node)

    def visit_StatementBlock(self, node: sast.StatementBlock):
        self._modify_typeinfo(node.operation_type)
        return self.generic_visit(node)

    def visit_IfBlock(self, node: sast.IfBlock):
        self._modify_typeinfo(node.operation_type)
        return self.generic_visit(node)

    def visit_ComputationBlock(self, node: sast.ComputationBlock):
        self._modify_typeinfo(node.operation_type)
        return self.generic_visit(node)

    def visit_Program(self, node: sast.Program):
        self._modify_typeinfo(node.operation_type)
        return self.generic_visit(node)
