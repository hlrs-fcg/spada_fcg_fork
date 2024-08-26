from dataclasses import dataclass
from typing import Sequence

import spatialstencil.syntax.stencil_ir.irnodes as sast
from spatialstencil.syntax.stencil_ir.irnodes import ComputationBlock


@dataclass
class ScopedUse:
    definition_scope: sast.ComputationBlock | sast.Program
    field_type: sast.FieldType

@dataclass
class ScopedDefinition:
    definition_scope: sast.ComputationBlock | sast.Program
    field_type: sast.FieldType


class DefUseAnalysis(sast.ScopedNodeVisitor):

    """
    For each field variable, constructs the uses of it,
    storing the field type object and the enclosing scope of it in a dictionary.

    Moreover, we construct for each use the defining identifier and the scope of the definition.
    This assumes that the program is in SSA form, that is, each identifier is defined exactly once.
    """
    def __init__(self,
                 def_use: dict[sast.Identifier, list[ScopedUse]] = None,
                 use_def: dict[sast.Identifier, ScopedDefinition] = None):
        """
        Initializes the def-use and use-def analysis.

        :param def_use: If None, the def-use analysis is disabled.
        :param use_def: If None, the use-def analysis is disabled.
        """
        super().__init__()
        self.def_use = def_use
        self.use_def = use_def
        self._current_scope = []

    def add_use(self, node: sast.Identifier, field_type: sast.FieldType):
        """
        Adds a use of a field to the def_use dictionary, with the current scope.

        :param node: The identifier
        :param field_type: The field type object
        """
        if self.def_use is None:
            return
        assert isinstance(node, sast.Identifier)

        if node not in self.def_use:
            self.def_use[node] = []
        self.def_use[node].append(ScopedUse(self.get_scope(), field_type))

    def add_definition(self, node: sast.Identifier, field_type: sast.FieldType):
        """
        Adds a definition of a field to the use_def dictionary, with the current scope.

        :param node: The identifier
        """
        if self.use_def is None:
            return
        assert isinstance(node, sast.Identifier)

        assert node not in self.use_def, f"Program is not in SSA form, {node} is defined multiple times"

        self.use_def[node] = ScopedDefinition(self.get_scope(), field_type)

    def visit_StatementBlock(self, node: sast.StatementBlock):
        """
        A statement uses all its argument types and defines all its outputs.

        :param node: The node to visit.
        """
        for arg_id, arg_t in zip(node.inputs, node.operation_type.source):
            self.add_use(arg_id, arg_t)
        for out_id, out_t in zip(node.outputs, node.operation_type.destination):
            self.add_definition(out_id, out_t)
        # WE INTENTIONALLY DO NOT RECURSE INTO THE STATEMENT BLOCK
        # AS IT IS A LOCAL SCOPE

    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        """
        A materialize operation uses its argument and defines its output.

        :param node: The node to visit.
        """
        self.add_use(node.value, node.operation_type.source[0])
        self.add_definition(node.result, node.operation_type.destination[0])
        self.generic_visit(node)


    def visit_IfBlock(self, node: sast.IfBlock):
        """
        An if block uses its condition and the conditions of its if-else blocks.
        The if-else blocks share the same type as the if block.
        An if block defines its outputs.

        :param node:  The node to visit.
        """
        self.add_use(node.condition, node.operation_type.source[0])

        for elif_block in node.else_ifs:
            if elif_block.condition:
                self.add_use(elif_block.condition, node.operation_type.source[0])

        for out_id, out_t in zip(node.outputs, node.operation_type.destination):
            self.add_definition(out_id, out_t)

        self.generic_visit(node)

    def pre_visit_ComputationBlock(self, node: sast.ComputationBlock):
        """
        A computation block uses all its inputs and defines all its outputs.

        :param node: The node to visit.
        """
        for arg_id, arg_t in zip(node.inputs, node.operation_type.source):
            self.add_use(arg_id, arg_t)
        for out_id, out_t in zip(node.outputs, node.operation_type.destination):
            self.add_definition(out_id, out_t)

    def do_visit_ComputationBlock(self, node: ComputationBlock):
        """
        Within the computation block, all inputs are defined.

        :param node: The node to visit.
        """
        for arg_id, arg_t in zip(node.inputs, node.operation_type.source):
            self.add_definition(arg_id, arg_t)
        self.generic_visit(node)

    def visit_ReturnOp(self, node: sast.ReturnOp):
        # A return uses all its arguments and does not define anything
        assert all(isinstance(arg.value, sast.Identifier) for arg in node.values)
        # And create a use for each field access
        for i in range(len(node.values)):

            arg, arg_t = node.values[i].value, node.operation_type.source[i]

            # TODO This is a hot-fix! Should be done elsewhere.
            if isinstance(arg_t, sast.AnyType):
                node.operation_type.source[i] = sast.FieldType.empty()
                arg_t = node.operation_type.source[i]

            assert isinstance(arg_t, sast.FieldType)
            self.add_use(arg, arg_t)
