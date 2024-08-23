from dataclasses import dataclass
from typing import Sequence

import spatialstencil.syntax.stencil_ir.irnodes as sast

@dataclass
class ScopedUse:
    definition_scope: sast.ComputationBlock | sast.Program
    field_type: sast.FieldType


class DefUseAnalysis(sast.ScopedNodeVisitor):

    """
    For each field variable, constructs the uses of it,
    storing the field type object and the enclosing scope of it in a dictionary.
    """

    def __init__(self, def_use: dict[sast.Identifier, list[ScopedUse]]):
        super().__init__()
        self.def_use = def_use
        self._current_scope = []

    def add(self, node: sast.Identifier, field_type: sast.FieldType):
        """
        Adds a use of a field to the def_use dictionary, with the current scope.
        :param node: The identifier
        :param field_type: The field type object
        :return:
        """
        assert isinstance(node, sast.Identifier)

        if node not in self.def_use:
            self.def_use[node] = []
        self.def_use[node].append(ScopedUse(self.get_scope(), field_type))

    def visit_StatementBlock(self, node: sast.StatementBlock):
        """
        A statement uses all its argument types
        :param node:
        :return:
        """
        for arg_id, arg_t in zip(node.inputs, node.operation_type.source):
            self.add(arg_id, arg_t)
        # WE INTENTIONALLY DO NOT RECURSE INTO THE STATEMENT BLOCK
        # AS IT IS A LOCAL SCOPE

    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        """
        A materialize operation uses its argument
        :param node:
        :return:
        """
        self.add(node.value, node.operation_type.source[0])
        self.generic_visit(node)


    def visit_IfBlock(self, node: sast.IfBlock):
        """
        An if block uses its condition and the conditions of its if-else blocks.
        The if-else blocks share the same type as the if block.
        :param node:
        :return:
        """
        self.add(node.condition, node.operation_type.source[0])

        for elif_block in node.else_ifs:
            if elif_block.condition:
                self.add(elif_block.condition, node.operation_type.source[0])

        self.generic_visit(node)

    def visit_ComputationBlock(self, node: sast.ComputationBlock):
        """
        A computation block uses all its inputs
        :param node:
        :return:
        """
        for arg_id, arg_t in zip(node.inputs, node.operation_type.source):
            self.add(arg_id, arg_t)
        super().visit_ComputationBlock(node)


    def visit_ReturnOp(self, node: sast.ReturnOp):
        # A return uses all its arguments
        assert all(isinstance(arg.value, sast.Identifier) for arg in node.values)
        # And create a use for each field access
        for i in range(len(node.values)):

            arg, arg_t = node.values[i].value, node.operation_type.source[i]

            # TODO This is a hot-fix! Should be done elsewhere.
            if isinstance(arg_t, sast.AnyType):
                node.operation_type.source[i] = sast.FieldType.empty()
                arg_t = node.operation_type.source[i]

            assert isinstance(arg_t, sast.FieldType)
            self.add(arg, arg_t)
