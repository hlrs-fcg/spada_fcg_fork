"""
Provides basic AST/IR utilities for traversal, transformation, and find/replacement.

Walking and transformation are based on Python's ``ast`` module, but introduce additional
functionality such as IR language testing and dataclass support.
"""
import ast
from dataclasses import dataclass
from typing import Generic, TypeVar


@dataclass
class BaseNode:
    """
    Base class for AST/IR nodes.
    """

    def iter_fields(self):
        """
        Yield a tuple of ``(fieldname, value)`` for each field in the dataclass.
        """
        for field in self.__dataclass_fields__:
            try:
                yield field, getattr(self, field)
            except AttributeError:
                pass

    def iter_child_nodes(self, ir_node_class: type['BaseNode'] = None):
        """
        Yield all direct child AST/IR nodes of node.
        """
        ir_node_class = ir_node_class or BaseNode
        for _, field in self.iter_fields():
            if isinstance(field, BaseNode):
                yield field
            elif isinstance(field, (tuple, list)):
                for item in field:
                    if isinstance(item, BaseNode):
                        yield item


# Create a generic type T that extends the base node type
BaseNodeT = TypeVar('BaseNodeT', bound=BaseNode)


def walk(node: BaseNode):
    """
    Recursively yield all descendant nodes in the tree starting at ``node``
    (including ``node`` itself), in breadth-first order. This function is
    based on ``ast.walk``.
    """
    from collections import deque
    todo = deque([node])
    while todo:
        node = todo.popleft()
        todo.extend(node.iter_child_nodes())
        yield node


class IRNodeVisitor(Generic[BaseNodeT]):
    """
    A node visitor base class that walks the AST/IR node tree and visits children
    using named functions. See ``ast.NodeVisitor`` for more information.

    The class also performs optional language validation checks by passing in the base IR node class.
    """

    def __init__(self, ir_node_class: type[BaseNodeT] = BaseNode):
        self.base_node = ir_node_class

    def _validate_node_type(self, node: BaseNodeT):
        if self.base_node is not BaseNode and isinstance(node, BaseNode) and not isinstance(node, self.base_node):
            raise TypeError(f'Node {node} does not match the IR language with base node {self.base_node}')

    def visit(self, node: BaseNodeT):
        """Visit a node."""
        self._validate_node_type(node)
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: BaseNodeT):
        """Called if no explicit visitor function exists for a node."""
        if isinstance(node, BaseNode):
            for _, value in node.iter_fields():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, BaseNode):
                            self.visit(item)
                elif isinstance(value, BaseNode):
                    self.visit(value)


class IRNodeTransformer(IRNodeVisitor[BaseNodeT]):
    """
    A ``NodeVisitor`` subclass that walks the AST/IR and allows modification of nodes.
    See ``ast.NodeTransformer`` for more information.
    """

    def generic_visit(self, node: BaseNodeT):
        for field, old_value in node.iter_fields():
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, BaseNode):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, BaseNode):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, BaseNode):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class ASTFindReplace(ast.NodeTransformer):
    """
    Finds and replaces a name with another value on Python ASTs.
    """

    def __init__(self, repldict: dict[str, ast.AST]):
        """
        Creates a find-and-replace AST node transformer.

        :param repldict: A dictionary mapping a source name to a target replacement AST node.
        """
        self.replace_count = 0
        self.repldict = repldict
        # If ast.Names were given, use them as keys as well
        self.repldict.update({k.id: v for k, v in self.repldict.items() if isinstance(k, ast.Name)})

    def visit_Name(self, node: ast.Name):
        if node.id in self.repldict:
            val = self.repldict[node.id]
            if isinstance(val, ast.AST):
                new_node = ast.copy_location(val, node)
            else:
                new_node = ast.copy_location(ast.parse(str(self.repldict[node.id])).body[0].value, node)
            self.replace_count += 1
            return new_node

        return self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword):
        if node.arg in self.repldict:
            val = self.repldict[node.arg]
            if isinstance(val, ast.AST):
                val = ast.unparse(val)
            node.arg = val
            self.replace_count += 1
        return self.generic_visit(node)
