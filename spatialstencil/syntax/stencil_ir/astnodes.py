"""
Native class definitions for the spatial stencil Abstract Syntax Tree (AST).
"""
from dataclasses import dataclass, field
import enum
from typing import Literal


class ScalarType(enum.Enum):
    UNKNOWN = enum.auto()  # Not yet type-inferred
    i8 = enum.auto()
    i16 = enum.auto()
    i32 = enum.auto()
    u8 = enum.auto()
    u16 = enum.auto()
    u32 = enum.auto()
    f16 = enum.auto()
    f32 = enum.auto()
    f64 = enum.auto()
    bool = enum.auto()


class ComputationType(enum.Enum):
    PARALLEL = 0
    FORWARD = 1
    BACKWARD = 2


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

    def as_ir(self, indent: int = 0) -> str:
        """
        Returns the AST node as a parseable Stencil IR version.

        :param indent: Indentation for the IR node.
        """
        return str(self)

    def validate(self) -> None:
        """
        Runs assertions on the node
        """
        pass


class Domain(Node):
    """
    Abstract domain type.
    """

    def as_ir(self, indent: int = 0) -> str:
        return 'spst.domain'


def _val_or_unk(val: int | None) -> str:
    """
    Helper function that prints out a value or a question mark if None.
    """
    if val is None:
        return '?'
    return str(val)


@dataclass
class DimTuple(Node):
    values: tuple[int | None]

    def as_ir(self, indent: int = 0) -> str:
        return f'({", ".join(_val_or_unk(v) for v in self.values)})'


@dataclass
class Cartesian(Domain):
    """
    Cartesian domain type in three dimensions.

    A None value for each dimension means "unknown" (or "?")
    """
    x: int | None
    y: int | None
    z: int | None

    def as_ir(self, indent: int = 0) -> str:
        return f'spst.cartesian<{_val_or_unk(self.x)}, {_val_or_unk(self.y)}, {_val_or_unk(self.z)}>'


@dataclass
class Extent(Node):
    """
    Extents of a field.
    """
    extents: list[DimTuple]

    def as_ir(self, indent: int = 0) -> str:
        return f'spst.extent<{", ".join(self.extents.as_ir())}>'


@dataclass
class FieldType(Node):
    domain: Domain
    extent: Extent
    dtype: ScalarType

    @classmethod
    def empty(cls) -> 'FieldType':
        """
        Creates an empty (not type/shape-inferred) field type.
        """
        return FieldType(
            domain=Cartesian(None, None, None), extent=Extent([DimTuple((None, None, None))]), dtype=ScalarType.UNKNOWN)

    def as_ir(self, indent: int = 0) -> str:
        return f'spst.field<{self.domain.as_ir()}, {self.extent.as_ir()}, {self.dtype.value}>'


@dataclass
class Interval(Node):
    start: int | Literal["?"] | None = "?"
    end: int | Literal["?"] | None = "?"

    def as_ir(self, indent: int = 0) -> str:
        return f'spst.interval<{self.start}, {self.end}>'


@dataclass
class StringLiteral(Node):
    """
    A string literal AST node (``"string"``).
    """
    value: str

    def as_ir(self, indent: int = 0) -> str:
        return f'"{self.value}"'


@dataclass
class Identifier(Node):
    """
    A field/scalar identifier (``%abc``).
    """
    name: str
    version: int = 0
    dtype: FieldType = field(default_factory=FieldType.empty)

    def validate(self) -> None:
        assert self.dtype.dtype != ScalarType.f64

    def as_ir(self, indent: int = 0) -> str:
        if self.version != 0:
            return f'%{self.name}#{self.version}'
        return f'%{self.name}'


@dataclass
class Constant(Node):
    """
    A constant literal
    """
    value: int | float

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.value}'


@dataclass
class UnaryOperator(Node):
    """
    A unary operator (+x, -x, ~x, not x)
    """
    op: str
    value: 'Expression'

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.op}{self.value.as_ir()}'


@dataclass
class BinaryOperator(Node):
    """
    A binary operator (one of: x {or, and, |, ^, &, >>, <<, +, -, *, /, %, >, <, ==, >=, <=, !=, **} y)
    """
    left: 'Expression'
    op: str
    right: 'Expression'

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.left.as_ir()} {self.op} {self.right.as_ir()}'


@dataclass
class TernaryOperator(Node):
    """
    A ternary operator (x if y else z)
    """
    true_value: 'Expression'
    test: 'Expression'
    false_value: 'Expression'

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.true_value.as_ir()} if ({self.test.as_ir()}) else {self.false_value.as_ir()}'


@dataclass
class Subscript(Node):
    """
    A field subscript (of the form %x[0, 1, 0])
    """
    value: Identifier
    subscript: tuple[int, int, int]

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.value.as_ir()}[{", ".join(self.subscript)}]'


# TODO(later): Call nodes (e.g., for math calls)


@dataclass
class Expression(Node):
    """
    An expression that can take the form of an identifier, literal, subscript, or a unary/binary/ternary operator.
    """
    value: Identifier | Constant | Subscript | UnaryOperator | BinaryOperator | TernaryOperator

    def as_ir(self, indent: int = 0) -> str:
        if isinstance(self.value, (Identifier, Constant, Subscript, UnaryOperator)):
            return self.value.as_ir(indent)
        return f'({self.value.as_ir(indent)})'


@dataclass
class Operation(Node):
    """
    Base class for an single operation.
    """
    pass


@dataclass
class Block(Node):
    """
    Base class for a block of operations.
    """
    pass


@dataclass
class ReturnOp(Operation):
    value: Identifier

    def validate(self) -> None:
        assert isinstance(self.value, Identifier)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return f'{indent_str}spst.return {self.value.as_ir()} : {self.value.dtype.as_ir()}'


@dataclass
class MaterializeOp(Operation):
    result: Identifier
    value: Identifier

    def validate(self) -> None:
        assert isinstance(self.result, Identifier)
        assert isinstance(self.value, Identifier)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return (f'{indent_str}{self.result.as_ir()} = spst.materialize ({self.value.as_ir()})'
                f' : {self.value.dtype.as_ir()} -> {self.result.dtype.as_ir()}')


@dataclass
class StatementBlock(Block):
    """
    A single statement in a Stencil IR computation.
    """
    output: Identifier
    inputs: list[Identifier]
    body: ReturnOp

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        inputs = ', '.join(i.as_ir() for i in self.inputs)
        input_types = ', '.join(i.dtype.as_ir() for i in self.inputs)
        output = self.output.as_ir()
        output_type = self.output.dtype.as_ir()
        result = f'{indent_str}{output} = spst.statement ({inputs})'
        result += f' : {input_types} -> {output_type} '
        result += '{\n'
        result += self.body.as_ir(indent + 1)
        #result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.body)
        result += '\n' + indent_str + '}'
        return result


@dataclass
class IfBlock(Block, Operation):
    """
    If/elif/else block operating on a mask tensor.
    """
    result: Identifier
    condition: Identifier
    body: list[Operation]
    else_ifs: list[tuple[Identifier, list[Operation]]] | None  # List of (condition, body)
    orelse: list[Operation] | None

    def validate(self) -> None:
        assert isinstance(self.result, Identifier)
        assert isinstance(self.condition, Identifier)
        if self.else_ifs:
            assert isinstance(self.else_ifs, list)
            assert all(isinstance(econd, Identifier) for econd, _ in self.else_ifs)

        # Check terminators
        assert isinstance(self.body[-1], ReturnOp)
        if self.else_ifs:
            assert all(isinstance(estmts[-1], ReturnOp) for _, estmts in self.else_ifs)
        if self.orelse:
            assert isinstance(self.orelse[-1], ReturnOp)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        result = f'{indent_str}{self.result.as_ir()} = spst.if ({self.condition.as_ir()})'
        result += f' : {self.condition.dtype.as_ir()} -> {self.result.dtype.as_ir()} '
        result += '{\n'
        result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.body)
        result += '\n' + indent_str + '}'
        if self.else_ifs:
            for elif_cond, elif_body in self.else_ifs:
                result += f' elif ({elif_cond.as_ir()}) '
                result += '{\n'
                result += '\n'.join(stmt.as_ir(indent + 1) for stmt in elif_body)
                result += '\n' + indent_str + '}'
        if self.orelse:
            result += ' else {\n'
            result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.orelse)
            result += '\n' + indent_str + '}'
        return result


@dataclass
class ComputationBlock(Block):
    """
    A computational block with an interval in a stencil IR computation.
    """
    outputs: list[Identifier]
    inputs: list[Identifier]
    schedule: ComputationType
    interval: tuple[Interval, Interval, Interval]
    body: list[StatementBlock | IfBlock | MaterializeOp | ReturnOp]

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        inputs = ', '.join(i.as_ir() for i in self.inputs)
        input_types = ', '.join(i.dtype.as_ir() for i in self.inputs)
        outputs = ', '.join(o.as_ir() for o in self.outputs)
        output_types = ', '.join(o.dtype.as_ir() for o in self.outputs)
        result = f'{indent_str}{outputs} = spst.computation ({inputs})'

        # Attributes
        result += '{\n'
        result += f'{indent_str} schedule = {self.schedule.value}\n'
        result += f'{indent_str} interval = [{", ".join(i.as_ir() for i in self.interval)}]\n'
        result += '\n' + indent_str + '}'
        # Types
        result += f' : {input_types} -> {output_types} '
        # Body
        result += '{\n'
        result += self.body.as_ir(indent + 1)
        #result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.body)
        result += '\n' + indent_str + '}'
        return result


@dataclass
class Program(Node):
    """
    Root node of a stencil program AST.
    """
    outputs: list[Identifier]
    name: str | None
    inputs: list[Identifier]
    computations: list[ComputationBlock]

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        newline = '\n'
        inputs = ', '.join(i.as_ir() for i in self.inputs)
        input_types = ', '.join(i.dtype.as_ir() for i in self.inputs)
        outputs = ', '.join(o.as_ir() for o in self.outputs)
        output_types = ', '.join(o.dtype.as_ir() for o in self.outputs)
        name = f' @{self.name}' if self.name else ''

        return (f'{indent_str}{outputs} = spst.program{name} ({inputs}) {{ }}'
                f' : {input_types} -> {output_types}'
                '{\n'
                f'{newline.join(c.as_ir(indent + 1) for c in self.computations)}'
                '\n' + indent_str + '}')
