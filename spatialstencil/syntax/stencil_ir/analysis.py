"""
Analysis passes on the Stencil IR.
"""
from collections import defaultdict
from spatialstencil.syntax.stencil_ir import irnodes as sast


class ExtentCollector(sast.NodeVisitor):
    """
    A node visitor that collects all input and output extents from field accesses in the visited blocks/statements.
    """

    def __init__(self):
        super().__init__()
        self.extents: dict[str, set[tuple[int]]] = defaultdict(set)

    def visit_Identifier(self, node: sast.Identifier):
        # If a bare identifier (i.e., no subscript) is used, the extent (0, 0, 0) should be added
        self.extents[node.name].add((0, 0, 0))

    def visit_Subscript(self, node: sast.Subscript):
        # If a subscript is found, add its subscript to the extents.
        # Make sure not to recursively visit into the subscript to avoid adding (0, 0, 0)
        self.extents[node.value.name].add(node.subscript)


def collect_extents(node: sast.Node) -> dict[str, set[tuple[int]]]:
    """
    Collects all input and output extents from field accesses in this block.
    """
    collector = ExtentCollector()
    collector.visit(node)
    return collector.extents


class InputOutputCollector(sast.NodeVisitor):
    """
    A node visitor that collects all input and output fields in the visited blocks/statements.
    """

    def __init__(self):
        super().__init__()
        self.inputs: set[sast.Identifier] = set()
        self.outputs: set[sast.Identifier] = set()

    # Visitors that return used identifiers
    def visit_Identifier(self, node: sast.Identifier) -> set[sast.Identifier]:
        return {node}

    def visit_Expression(self, node: sast.Expression) -> set[sast.Identifier]:
        if isinstance(node.value, (int, float)):
            return set()
        return self.visit(node.value)

    def visit_Subscript(self, node: sast.Subscript) -> set[sast.Identifier]:
        return self.visit(node.value)

    def visit_UnaryOperator(self, node: sast.UnaryOperator) -> set[sast.Identifier]:
        return self.visit(node.value)

    def visit_BinaryOperator(self, node: sast.BinaryOperator) -> set[sast.Identifier]:
        return self.visit(node.left) | self.visit(node.right)

    def visit_TernaryOperator(self, node: sast.TernaryOperator) -> set[sast.Identifier]:
        return self.visit(node.test) | self.visit(node.true_value) | self.visit(node.false_value)

    def visit_MathCall(self, node: sast.MathCall) -> list[sast.Identifier]:
        return set().union(*(self.visit(arg) for arg in node.arguments))

    # Visitors that infer input/output status from returned identifiers
    def visit_ReturnOp(self, node: sast.ReturnOp):
        self.inputs |= set().union(*(self.visit(retval) for retval in node.values))

    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        self.inputs |= self.visit(node.value)
        self.outputs |= self.visit(node.result)

    def visit_AssignOp(self, node: sast.AssignOp):
        self.inputs |= self.visit(node.value)
        # self.outputs |= self.visit(node.result)  # Only include return values as outputs

    def visit_StatementBlock(self, node: sast.StatementBlock):
        self.inputs |= set(node.inputs)
        self.outputs |= set(node.outputs)

    def visit_IfBlock(self, node: sast.IfBlock):
        self.inputs |= self.visit(node.condition)
        if node.else_ifs:
            for elif_cond, _ in node.else_ifs:
                self.inputs |= self.visit(elif_cond)

        self.outputs |= set(node.outputs)

        # Recurse to children
        self.generic_visit(node)
