import unittest
from dataclasses import dataclass
from spatialstencil.syntax.helpers import BaseNode, IRNodeTransformer, IRNodeVisitor


@dataclass
class SimpleNode(BaseNode):
    a: int
    f: int


@dataclass
class SequenceNode(BaseNode):
    f: int
    tup: tuple[BaseNode, BaseNode]
    lst: list[BaseNode]
    lst_of_tuples: list[tuple[BaseNode, BaseNode]]


# Create a IRNodeVisitor that visits all nodes and adds the value of all integer fields
class SumVisitor(IRNodeVisitor):
    def __init__(self):
        super().__init__(BaseNode)
        self.sum = 0

    def visit_SimpleNode(self, node: SimpleNode):
        self.sum += node.a
        self.sum += node.f
        self.generic_visit(node)

    def visit_SequenceNode(self, node: SequenceNode):
        self.sum += node.f
        self.generic_visit(node)


# Create a transformer that adds one to every integer field
class IncrementVisitor(IRNodeTransformer):
    def visit_SimpleNode(self, node: SimpleNode):
        node.a += 1
        node.f += 1
        return node

    def visit_SequenceNode(self, node: SequenceNode):
        node.f += 1
        node = self.generic_visit(node)
        return node


class TestIRNode(unittest.TestCase):
    def test_ir_node_transformer_identity(self):
        # Create a sequence node
        node = SequenceNode(1, (SimpleNode(1, 1), SimpleNode(2, 6)), [SimpleNode(3, 6), SimpleNode(4, 8)], [(SimpleNode(1, 1), SimpleNode(2, 6)), (SimpleNode(3, 6), SimpleNode(4, 8))])
        # Create a copy
        cp_node = SequenceNode(1, (SimpleNode(1, 1), SimpleNode(2, 6)), [SimpleNode(3, 6), SimpleNode(4, 8)], [(SimpleNode(1, 1), SimpleNode(2, 6)), (SimpleNode(3, 6), SimpleNode(4, 8))])
        # Create a node transformer
        transformer = IRNodeTransformer()
        # Transform the node
        transformer.visit(node)
        # Check it is the same
        self.assertEqual(node, cp_node)

    def test_ir_node_transformer_increment(self):
        node = SequenceNode(1, (SimpleNode(1, 1), SimpleNode(2, 6)), [SimpleNode(3, 6), SimpleNode(4, 8)], [(SimpleNode(1, 1), SimpleNode(2, 6)), (SimpleNode(3, 6), SimpleNode(0, 0))])
        golden = SequenceNode(2, (SimpleNode(2, 2), SimpleNode(3, 7)), [SimpleNode(4, 7), SimpleNode(5, 9)], [(SimpleNode(2, 2), SimpleNode(3, 7)), (SimpleNode(4, 7), SimpleNode(1, 1))])

        transformer = IncrementVisitor()
        transformer.visit(node)

        self.assertEqual(node, golden)

    def test_ir_node_visitor(self):
        # Create a sequence node
        node = SequenceNode(1, (SimpleNode(1, 1), SimpleNode(2, 6)), [SimpleNode(3, 6), SimpleNode(4, 8)], [(SimpleNode(0, 1), SimpleNode(1, 0)), (SimpleNode(1, 1), SimpleNode(2, 0))])
        # Create a visitor
        visitor = SumVisitor()
        # Visit the node
        visitor.visit(node)
        # expected result = 1 + 1 + 1 + 2 + 6 + 3 + 6 + 4 + 8 + 0 + 1 + 1 + 0 + 1 + 1 + 2 + 0 = 38
        # Check the sum
        self.assertEqual(visitor.sum, 38)


if __name__ == '__main__':
    unittest.main()
