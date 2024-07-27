"""
Native class definitions for the spatial stencil Abstract Syntax Tree (AST).
"""
from dataclasses import dataclass


class Node:
    """
    Abstract class representing an AST node for spatial stencils.
    """

    @classmethod
    def from_lark(cls, args):
        """
        Simple constructor that calls the AST node object constructor with the
        AST children in order. See ``lark_to_ast.py`` for usage.
        """
        return cls(*args)


class Program(Node):
    """
    Root node of a stencil program AST.
    """
    pass


@dataclass
class StringLiteral(Node):
    """
    A string literal AST node (``"string"``).
    """
    value: str


@dataclass
class Identifier(Node):
    """
    A field/scalar identifier (``%abc``).
    """
    name: str
