"""
Native class definitions for the spatial stencil Intermediate Representation (IR).
"""
from dataclasses import dataclass, field
import enum
from typing import Literal

from spatialstencil.syntax.common.basenode import BaseNode
from spatialstencil.syntax.common import visitor


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


BIT_WIDTH = {
    ScalarType.UNKNOWN: 0,
    ScalarType.i8: 8,
    ScalarType.i16: 16,
    ScalarType.i32: 32,
    ScalarType.u8: 8,
    ScalarType.u16: 16,
    ScalarType.u32: 32,
    ScalarType.f16: 16,
    ScalarType.f32: 32,
    ScalarType.f64: 64,
    ScalarType.bool: 1,
}


class ComputationType(enum.Enum):
    # We are using numbers to ensure compatibility with GT4Py's AST values
    PARALLEL = 0
    FORWARD = 1
    BACKWARD = 2


class IRType:
    """
    Interface that indicates this node represents a type.
    """
    pass


class Node(BaseNode):
    """
    Abstract class representing an IR node for spatial stencils.
    """

    @classmethod
    def from_lark(cls, args):
        """
        Simple constructor that calls the IR node object constructor with the
        IR children in order. See ``lark_to_ast.py`` for usage.
        """
        return cls(*args)

    def as_ir(self, indent: int = 0) -> str:
        """
        Returns the node as a parseable Stencil IR version.

        :param indent: Indentation for the IR node.
        """
        return str(self)

    def validate(self) -> None:
        """
        Runs assertions on the node.
        """
        pass

    def __post_init__(self):
        self.validate()


class Domain(Node, IRType):
    """
    Abstract domain type.
    """

    def as_ir(self, indent: int = 0) -> str:
        return 'spst.domain'

    def is_unknown(self) -> bool:
        raise NotImplementedError('Abstract class. Method implemented in subclasses')


def _val_or_unk(val: int | None) -> str:
    """
    Helper function that prints out a value or a question mark if None.
    """
    if val is None:
        return '?'
    return str(val)




@dataclass
class OffsetAndInterval(Node):
    """
    Dimension tuple containing an offset and an interval of an ``Extent``.
    """
    # NOTE: We are using variable-length tuples here so that this type is hashable.
    values: tuple[int | None]
    interval: tuple[int | None] = field(default_factory=lambda: (0, None, 0, None, 0, None))

    def validate(self) -> None:
        # For every dimension, an interval has a start and end point
        assert len(self.interval) == 2 * len(self.values)

    def as_ir(self, indent: int = 0) -> str:
        output = f'({", ".join(_val_or_unk(v) for v in self.values)})'
        if self.interval != (0, None, 0, None, 0, None):
            interval_str = [f'{start}:{end}' for start, end in zip(self.interval[::2], self.interval[1::2])]
            output += f' in [{", ".join(interval_str)}]'
        return output


@dataclass
class Extent(Node, IRType):
    """
    Extents of a field.
    """
    extents: list[OffsetAndInterval]

    def as_ir(self, indent: int = 0) -> str:
        return "{" + f'{", ".join(e.as_ir() for e in self.extents)}' + "}"

    def is_unknown(self) -> bool:
        return all(dim is None for extent in self.extents for dim in extent.values)


@dataclass
class FieldType(Node, IRType):
    domain: Domain
    extent: Extent
    dtype: ScalarType

    @classmethod
    def empty(cls) -> 'FieldType':
        """
        Creates an empty (not type/shape-inferred) field type.
        """
        return FieldType(
            domain=Cartesian(Interval(), Interval(), Interval()),
            extent=Extent([OffsetAndInterval((None, None, None))]),
            dtype=ScalarType.UNKNOWN)

    def validate(self) -> None:
        assert self.dtype != ScalarType.f64

    def as_ir(self, indent: int = 0) -> str:
        return f'spst.field<{self.domain.as_ir()}, {self.extent.as_ir()}, {self.dtype.as_ir()}>'


DataType = FieldType | ScalarType


@dataclass
class OperationType(Node):
    source: list[DataType] = field(default_factory=lambda: [FieldType.empty()])
    destination: list[DataType] | None = None

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
class Interval(Node, IRType):
    # NOTE: We are using "?" to represent unknown values
    # NOTE: The None value can represent the limit relative to a known upper bound.
    start: int | Literal["?"] | None = "?"
    end: int | Literal["?"] | None = "?"

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.start}:{self.end}'
        #return f'spst.interval<{self.start}, {self.end}>'

    def is_unknown(self) -> bool:
        return self.start is None or self.end is None or self.start == "?" or self.end == "?"

    def union(self, other: 'Interval'):
        """
        Returns the union of two intervals.
        If one value is unknown, the other value is returned.
        TODO Check?
        Assumes none of the values is None.
        """
        assert self.start is not None
        assert self.end is not None
        assert other.start is not None
        assert other.end is not None

        if other.start == "?":
            start = self.start
        elif self.start == "?":
            start = other.start
        else:
            start = min(self.start, other.start)

        if other.end == "?":
            end = self.end
        elif self.end == "?":
            end = other.end
        else:
            end = max(self.end, other.end)

        return Interval(start, end)

    def intersect(self, other: 'Interval'):
        """
        Returns the intersection of two intervals.
        :param other:
        :return:
        """
        assert self.start is not None
        assert self.end is not None
        assert other.start is not None
        assert other.end is not None

        if other.start == "?":
            start = self.start
        elif self.start == "?":
            start = other.start
        else:
            start = max(self.start, other.start)

        if other.end == "?":
            end = self.end
        elif self.end == "?":
            end = other.end
        else:
            end = min(self.end, other.end)

        return Interval(start, end)


@dataclass
class Cartesian(Domain):
    """
    Cartesian domain type in three dimensions.

    A None value for each dimension means "unknown" (or "?")
    """
    x: Interval = field(default_factory=lambda: Interval())
    y: Interval = field(default_factory=lambda: Interval())
    z: Interval = field(default_factory=lambda: Interval())

    def as_ir(self, indent: int = 0) -> str:
        return f'[{self.x.as_ir()}, {self.y.as_ir()}, {self.z.as_ir()}]'
        #return f'spst.cartesian<{self.x.as_ir()}, {self.y.as_ir()}, {self.z.as_ir()}>'

    def is_unknown(self) -> bool:
        return self.x.is_unknown() or self.y.is_unknown() or self.z.is_unknown()

    def validate(self) -> None:
        # Domain must be fully defined
        assert isinstance(self.x, Interval)
        assert isinstance(self.y, Interval)
        assert isinstance(self.z, Interval)
        assert self.x.start is not None
        assert self.x.end is not None
        assert self.y.start is not None
        assert self.y.end is not None
        assert self.z.start is not None
        assert self.z.end is not None

    def union(self, other: 'Cartesian'):
        """
        Returns a new Cartesian domain that is the union of all intervals in the domain.
        Unknown values are replaced by the other domain's values.
        """
        return Cartesian(self.x.union(other.x), self.y.union(other.y), self.z.union(other.z))

    def intersect(self, other: 'Cartesian'):
        """
        Returns a new Cartesian domain that is the intersection of all intervals in the domain.
        Unknown values are replaced by the other domain's values.
        """
        return Cartesian(self.x.intersect(other.x), self.y.intersect(other.y), self.z.intersect(other.z))

    @staticmethod
    def from_tuple(tup: tuple) -> 'Cartesian':
        """
        Creates a Cartesian domain from a 6-tuple of integers or "?".
        """
        return Cartesian(Interval(tup[0], tup[1]), Interval(tup[2], tup[3]), Interval(tup[4], tup[5]))

    def intersect_with_ranges(self, tup: tuple[int | None]) -> 'Cartesian':
        """
        Creates a Cartesian sub-domain from a 6-tuple of intervals
        that may indicate a sub-domain of the original Cartesian domain
        through the use of negative values to indicate an offset from the
        upper bound of the domain and None to indicate the upper bound of the domain.
        """
        output_upperbound = [self.x.end, self.y.end, self.z.end]

        clean_interval_tuple = []
        for i in range(3):
            if tup[2 * i] is None:
                clean_interval_tuple.append(0)
            elif tup[2 * i] < 0:
                clean_interval_tuple.append(output_upperbound[i] + tup[2 * i])
            else:
                clean_interval_tuple.append(tup[2 * i])

            if tup[2 * i + 1] is None:
                clean_interval_tuple.append(output_upperbound[i])
            elif tup[2 * i + 1] < 0:
                clean_interval_tuple.append(output_upperbound[i] + tup[2 * i + 1])
            else:
                clean_interval_tuple.append(tup[2 * i + 1])

        domain = Cartesian.from_tuple(clean_interval_tuple)
        return domain

    def add(self, tuple) -> 'Cartesian':
        # Adds the value of the tuple to the Cartesian domain in each dimension
        assert len(tuple) == 3
        assert all(x is not None for x in tuple)
        assert not self.is_unknown()
        return Cartesian(Interval(self.x.start + tuple[0], self.x.end + tuple[0]),
                         Interval(self.y.start + tuple[1], self.y.end + tuple[1]),
                         Interval(self.z.start + tuple[2], self.z.end + tuple[2]))


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

    def __hash__(self):
        return hash((self.name, self.version))

    def __lt__(self, other: 'Identifier'):
        if self.name < other.name:
            return True
        if self.name == other.name:
            return self.version < other.version
        return False

    def __gt__(self, other: 'Identifier'):
        if self.name > other.name:
            return True
        if self.name == other.name:
            return self.version > other.version
        return False

    def __le__(self, other: 'Identifier'):
        return self < other or self == other

    def __ge__(self, other: 'Identifier'):
        return self > other or self == other


@dataclass
class UnaryOperator(Node):
    """
    A unary operator (+x, -x, ~x, not x)
    """
    op: str
    value: 'Expression'

    def validate(self) -> None:
        assert self.op in ('+', '-', '~', 'not')

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

    def validate(self) -> None:
        assert self.op in ('or', 'and', '|', '^', '&', '>>', '<<', '+', '-', '*', '/', '%', '>', '<', '==', '>=', '<=',
                           '!=', '**')

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
        return f'{self.true_value.as_ir()} if {self.test.as_ir()} else {self.false_value.as_ir()}'


@dataclass
class Subscript(Node):
    """
    A field subscript (of the form %x[0, 1, 0])
    """
    value: Identifier
    subscript: list[int]

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.value.as_ir()}[{", ".join(str(s) for s in self.subscript)}]'


@dataclass
class MathCall(Node):
    """
    A mathematical function call operation for stateless calls
    """
    func: str
    arguments: list['Expression']

    def validate(self) -> None:
        assert self.func in {'sqrt', 'cbrt'}

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.func}({", ".join(a.as_ir() for a in self.arguments)})'


@dataclass
class Expression(Node):
    """
    An expression that can take the form of an identifier, literal, subscript, or a unary/binary/ternary operator.
    """
    value: Identifier | int | float | Subscript | UnaryOperator | BinaryOperator | TernaryOperator | MathCall

    def as_ir(self, indent: int = 0) -> str:
        if isinstance(self.value, (int, float)):
            return str(self.value)
        if isinstance(self.value, (Identifier, Subscript, UnaryOperator, MathCall)):
            return self.value.as_ir(indent)
        return f'({self.value.as_ir(indent)})'


class Operation:
    """
    Interface for an operation.

    An Operation *must* have a field called ``operation_type`` defined.
    """
    operation_type: OperationType


class Block:
    """
    Interface for a block of operations.
    """
    body: list[Node]


@dataclass
class ReturnOp(Node, Operation):
    values: list[Expression]
    operation_type: OperationType = field(default_factory=lambda: OperationType([ScalarType.UNKNOWN]))

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return (f'{indent_str}spst.return {", ".join(v.as_ir() for v in self.values)}'
                f' : {self.operation_type.as_ir()}')


@dataclass
class MaterializeOp(Node, Operation):
    result: Identifier
    value: Identifier
    operation_type: OperationType = field(default_factory=OperationType)

    def validate(self) -> None:
        assert isinstance(self.result, Identifier)
        assert isinstance(self.value, Identifier)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return (f'{indent_str}{self.result.as_ir()} = spst.materialize ({self.value.as_ir()})'
                f' : {self.operation_type.as_ir()}')


@dataclass
class AssignOp(Node, Operation):
    result: Identifier
    value: Expression
    operation_type: OperationType = field(
        default_factory=lambda: OperationType([ScalarType.UNKNOWN], [ScalarType.UNKNOWN]))

    def validate(self) -> None:
        assert isinstance(self.result, Identifier)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return (f'{indent_str}{self.result.as_ir()} = {self.value.as_ir()}'
                f' : {self.operation_type.as_ir()}')


@dataclass
class Attribute(Node):
    name: str
    attr: Node

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.name} = {self.attr.as_ir()}'


@dataclass
class StatementBlock(Node, Operation, Block):
    """
    A single statement in a Stencil IR computation.
    """
    outputs: list[Identifier]
    inputs: list[Identifier]
    attributes: list[Attribute]
    operation_type: OperationType
    body: list[AssignOp | ReturnOp]

    def validate(self) -> None:
        assert isinstance(self.body[-1], ReturnOp)
        assert len(self.body[-1].values) == len(self.outputs)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        inputs = ', '.join(i.as_ir() for i in self.inputs)
        outputs = ', '.join(o.as_ir() for o in self.outputs)
        result = f'{indent_str}{outputs} = spst.statement ({inputs})'
        result += ' {}'
        result += f' : {self.operation_type.as_ir()} '
        result += '{\n'
        result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.body)
        result += '\n' + indent_str + '}'
        return result


@dataclass
class ElseIfBlock(Node, Block):
    """
    A single "else if" block. If condition is None, represents an "else" block
    """
    condition: Identifier | None
    body: list[StatementBlock | ReturnOp]

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        if self.condition is None:
            result = ' else '
        else:
            result = f' elif ({self.condition.as_ir()}) '

        result += '{\n'
        result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.body)
        result += '\n' + indent_str + '}'

        return result


@dataclass
class IfBlock(Node, Operation, Block):
    """
    If/elif/else block operating on a mask tensor.
    """
    outputs: list[Identifier]
    condition: Identifier
    operation_type: OperationType
    body: list[StatementBlock | ReturnOp]
    else_ifs: list[ElseIfBlock]

    def validate(self) -> None:
        assert isinstance(self.outputs, list)
        assert all(isinstance(r, Identifier) for r in self.outputs)
        assert isinstance(self.condition, Identifier)
        assert isinstance(self.else_ifs, list)
        assert all(isinstance(econd, ElseIfBlock) for econd in self.else_ifs)

        # Check terminators
        assert isinstance(self.body[-1], ReturnOp)
        if self.else_ifs:
            assert all(isinstance(estmts.body[-1], ReturnOp) for estmts in self.else_ifs)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        result = f'{indent_str}{", ".join(res.as_ir() for res in self.outputs)} = spst.if ({self.condition.as_ir()})'
        result += f' : {self.operation_type.as_ir()} '
        result += '{\n'
        result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.body)
        result += '\n' + indent_str + '}'
        for else_if in self.else_ifs:
            result += else_if.as_ir(indent)
        return result


@dataclass
class ComputationBlock(Node, Operation, Block):
    """
    A computational block with an interval in a stencil IR computation.
    """
    outputs: list[Identifier]
    inputs: list[Identifier]
    schedule: ComputationType
    interval: list[Interval]
    operation_type: OperationType
    body: list[StatementBlock | IfBlock | MaterializeOp]

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        inputs = ', '.join(i.as_ir() for i in self.inputs)
        outputs = ', '.join(o.as_ir() for o in self.outputs)
        result = f'{indent_str}{outputs} = spst.computation ({inputs}) '

        # Attributes
        result += '{\n'
        result += f'{indent_str} schedule = {self.schedule.name},\n'
        result += f'{indent_str} interval = ({", ".join(i.as_ir() for i in self.interval)})\n'
        result += indent_str + '}'
        # Types
        result += f' : {self.operation_type.as_ir()} '
        # Body
        result += '{\n'
        result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.body)
        result += '\n' + indent_str + '}'
        return result


@dataclass
class Program(Node, Operation, Block):
    """
    Root node of a stencil program AST.
    """
    outputs: list[Identifier]
    name: str | None
    inputs: list[Identifier]
    attributes: list[Attribute]
    operation_type: OperationType
    computations: list[ComputationBlock]

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        newline = '\n'
        inputs = ', '.join(i.as_ir() for i in self.inputs)
        outputs = ', '.join(o.as_ir() for o in self.outputs)
        name = f' @{self.name}' if self.name else ''

        return (f'{indent_str}{outputs} = spst.program{name}({inputs}) {{}}'
                f' : {self.operation_type.as_ir()} '
                '{\n'
                f'{newline.join(c.as_ir(indent + 1) for c in self.computations)}'
                '\n' + indent_str + '}')


class NodeVisitor(visitor.IRNodeVisitor[Node]):

    def __init__(self, *args, **kwargs):
        super().__init__(Node, *args, **kwargs)


class NodeTransformer(visitor.IRNodeTransformer[Node]):

    def __init__(self, *args, **kwargs):
        super().__init__(Node, *args, **kwargs)
