"""
Native class definitions for the spatial stencil Abstract Syntax Tree (AST).
"""
from dataclasses import dataclass


class Node:
    """
    Abstract class representing an AST node for spatial stencils.
    """
    pass


class Program(Node):
    """
    Root node of a stencil program AST.
    """
    pass


@dataclass
class StringLiteral(Node):
    """
    A string literal AST node ("string").
    """
    value: str
