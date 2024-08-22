from dataclasses import dataclass
from typing import Sequence

import spatialstencil.syntax.stencil_ir.irnodes as sast

@dataclass
class ScopedUse:
    definition_scope: sast.ComputationBlock | sast.Program
    field_type: sast.FieldType


class DefUseAnalysis(sast.NodeVisitor):

    """
    For each field variable, constructs the uses of it,
    storing the field type object and the enclosing scope of it in a dictionary.
    """

    _current_scope: list[sast.ComputationBlock | sast.Program | None]

    def get_scope(self):
        return self._current_scope[-1]

    def push_scope(self, scope):
        self._current_scope.append(scope)

    def pop_scope(self):
        self._current_scope.pop()

    def __init__(self, def_use: dict[sast.Identifier, list[ScopedUse]]):
        super().__init__()
        self.def_use = def_use
        self._current_scope = []

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

    def visit_MathCall(self, node: sast.MathCall):
        """
        A math call implies an access at 0, 0, 0. So we should add it!
        :param node:
        :return:
        """
        raise NotImplementedError("Math calls are not supported yet")

    def visit_IfBlock(self, node: sast.IfBlock):
        """
        An if block uses its condition and the conditions of its if-else blocks.
        The if-else blocks share the same type as the if block.
        :param node:
        :return:
        """
        self.add(node.condition, node.operation_type.source[0])

        for elif_block in node.else_ifs:
            self.add(elif_block.condition, node.operation_type.source[0])

        self.generic_visit(node)

    def add(self, node: sast.Identifier, field_type: sast.FieldType):
        """
        Adds a use of a field to the def_use dictionary, with the current scope.
        :param node: The identifier
        :param field_type: The field type object
        :return:
        """
        if node not in self.def_use:
            self.def_use[node] = []
        self.def_use[node].append(ScopedUse(self.get_scope(), field_type))

    def visit_ComputationBlock(self, node: sast.ComputationBlock):
        """
        A computation block uses all its inputs
        :param node:
        :return:
        """
        for arg_id, arg_t in zip(node.inputs, node.operation_type.source):
            self.add(arg_id, arg_t)
        self.push_scope(node)
        self.generic_visit(node)
        self.pop_scope()

    def visit_Program(self, node: sast.Program):
        # A program pushes a new scope, we currently do not consider inputs as uses
        self.push_scope(node)
        self.generic_visit(node)
        self.pop_scope()

    @staticmethod
    def _artificial_type(offset: Sequence[int], scalar_type: sast.ScalarType):
        return sast.FieldType(sast.Domain(),
                              sast.Extent([sast.Offset((offset[0], offset[1], offset[2]))]),
                              scalar_type)

    #def visit_Subscript(self, node: sast.Subscript):
        ###self.add(node.value, self._artificial_type(node.subscript))
        # Do not visit the identifier

    def visit_ReturnOp(self, node: sast.ReturnOp):
        # A return uses all its arguments
        # For now, we do not use the return value as a use
        assert all(isinstance(arg.value, sast.Identifier) for arg in node.values)

        # And create a use for each field access
        for arg, arg_t in zip(node.values, node.operation_type.source):
            # If it's an identifier, directly add the use
            if isinstance(arg.value, sast.Identifier):
                if isinstance(arg_t, sast.ScalarType):
                    self.add(arg.value, sast.FieldType(sast.Domain(),
                                                       sast.Extent([sast.Offset((0, 0, 0))]),
                                                       arg_t))
                else:
                    self.add(arg.value, arg_t)
