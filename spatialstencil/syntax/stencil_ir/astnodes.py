"""
Native class definitions for the spatial stencil Abstract Syntax Tree (AST).
"""
from dataclasses import dataclass, field
import enum
from typing import Literal
import pprint


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

    def as_ir(self, indent: int = 0) -> str:
        return self.name


class ComputationType(enum.Enum):
    # We are using numbers to ensure compatibility with GT4Py's AST values
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
        Runs assertions on the node.
        """
        pass

    def pretty(self) -> str:
        """
        Pretty-prints the contents of this AST node.
        """
        return pprint.pformat(self)


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
        return f'spst.extent<{", ".join(e.as_ir() for e in self.extents)}>'


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

    def validate(self) -> None:
        assert self.dtype != ScalarType.f64

    def as_ir(self, indent: int = 0) -> str:
        return f'spst.field<{self.domain.as_ir()}, {self.extent.as_ir()}, {self.dtype.as_ir()}>'


@dataclass
class TypeInfo(Node):
    source: FieldType | list[FieldType] = field(default_factory=FieldType.empty)
    destination: FieldType | list[FieldType] | None = None

    def as_ir(self, indent: int = 0) -> str:
        if isinstance(self.source, list):
            srcstr = ", ".join(v.as_ir() for v in self.source)
        else:
            srcstr = self.source.as_ir()

        if self.destination is None:
            return srcstr

        if isinstance(self.destination, list):
            dststr = ", ".join(v.as_ir() for v in self.destination)
        else:
            dststr = self.destination.as_ir()

        return f'{srcstr} -> {dststr}'


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

    def as_ir(self, indent: int = 0) -> str:
        if self.version != 0:
            return f'%{self.name}#{self.version}'
        return f'%{self.name}'


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
        return f'{self.value.as_ir()}[{", ".join(str(s) for s in self.subscript)}]'


# TODO(later): Call nodes (e.g., for math calls)


@dataclass
class Expression(Node):
    """
    An expression that can take the form of an identifier, literal, subscript, or a unary/binary/ternary operator.
    """
    value: Identifier | int | float | Subscript | UnaryOperator | BinaryOperator | TernaryOperator

    def as_ir(self, indent: int = 0) -> str:
        if isinstance(self.value, (int, float)):
            return str(self.value)
        if isinstance(self.value, (Identifier, Subscript, UnaryOperator)):
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
    values: list[Expression]
    typeinfo: TypeInfo = field(default_factory=TypeInfo)

    def validate(self) -> None:
        assert isinstance(self.values[0].value, Identifier)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return (f'{indent_str}spst.return {", ".join(v.as_ir() for v in self.values)}'
                f' : {self.typeinfo.as_ir()}')


@dataclass
class MaterializeOp(Operation):
    result: Identifier
    value: Identifier
    typeinfo: TypeInfo = field(default_factory=TypeInfo)

    def validate(self) -> None:
        assert isinstance(self.result, Identifier)
        assert isinstance(self.value, Identifier)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return (f'{indent_str}{self.result.as_ir()} = spst.materialize ({self.value.as_ir()})'
                f' : {self.typeinfo.as_ir()}')


@dataclass
class AssignOp(Operation):
    result: Identifier
    value: Expression
    typeinfo: TypeInfo = field(default_factory=TypeInfo)

    def validate(self) -> None:
        assert isinstance(self.result, Identifier)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return (f'{indent_str}{self.result.as_ir()} = {self.value.as_ir()}'
                f' : {self.typeinfo.as_ir()}')


@dataclass
class StatementBlock(Block):
    """
    A single statement in a Stencil IR computation.
    """
    output: Identifier
    inputs: list[Identifier]
    attributes: dict[str, Node]
    typeinfo: TypeInfo
    body: list[AssignOp | ReturnOp]

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        inputs = ', '.join(i.as_ir() for i in self.inputs)
        output = self.output.as_ir()
        result = f'{indent_str}{output} = spst.statement ({inputs})'
        result += ' {}'
        result += f' : {self.typeinfo.as_ir()} '
        result += '{\n'
        result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.body)
        result += '\n' + indent_str + '}'
        return result


@dataclass
class IfBlock(Block, Operation):
    """
    If/elif/else block operating on a mask tensor.
    """
    result: Identifier
    condition: Identifier
    typeinfo: TypeInfo
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
        result += f' : {self.typeinfo.as_ir()} '
        result += '{\n'
        result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.body)
        result += indent_str + '}'
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
    typeinfo: TypeInfo
    body: list[StatementBlock | IfBlock | MaterializeOp | ReturnOp]

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        inputs = ', '.join(i.as_ir() for i in self.inputs)
        outputs = ', '.join(o.as_ir() for o in self.outputs)
        result = f'{indent_str}{outputs} = spst.computation ({inputs}) '

        # Attributes
        result += '{\n'
        result += f'{indent_str} schedule = {self.schedule.name},\n'
        result += f'{indent_str} interval = [{", ".join(i.as_ir() for i in self.interval)}]\n'
        result += indent_str + '}'
        # Types
        result += f' : {self.typeinfo.as_ir()} '
        # Body
        result += '{\n'
        result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.body)
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
    attributes: dict[str, Node]
    typeinfo: TypeInfo
    computations: list[ComputationBlock]

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        newline = '\n'
        inputs = ', '.join(i.as_ir() for i in self.inputs)
        outputs = ', '.join(o.as_ir() for o in self.outputs)
        name = f' @{self.name}' if self.name else ''

        return (f'{indent_str}{outputs} = spst.program{name}({inputs}) {{}}'
                f' : {self.typeinfo.as_ir()} '
                '{\n'
                f'{newline.join(c.as_ir(indent + 1) for c in self.computations)}'
                '\n' + indent_str + '}')
