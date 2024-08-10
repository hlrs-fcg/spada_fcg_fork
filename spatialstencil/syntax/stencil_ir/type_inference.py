"""
Contains type/extent inference functionality for the Stencil IR.
"""
from spatialstencil.syntax.stencil_ir import irnodes as sast
from spatialstencil.syntax.stencil_ir import analysis
from spatialstencil.syntax import helpers
import copy
import itertools
import warnings


def infer_types(program: sast.Program,
                default_float_dtype: sast.ScalarType = sast.ScalarType.f32,
                default_int_dtype: sast.ScalarType = sast.ScalarType.i32,
                domain: tuple[int] | None = None):
    """
    Infers all types in a Stencil IR program with optional domain size or halo extents.
    If domain size is not given, shapes will remain unknown ("?"). If halo is not given,
    zero halo extents are assumed.

    Operates in-place on the ``Program`` object.

    :param program: The root AST node of the Stencil IR program.
    :param default_float_dtype: The float type to use for float literals (e.g. 0.0) and fields that do not have an
                                explicit type.
    :param default_int_dtype: The integer type to use for integer literals and integral fields that do not have an
                              explicit type.
    :param domain: An optional 3-tuple representing domain size (x, y, z).
    """
    infer_inputs_and_outputs(program)
    infer_scalar_types(program, default_float_dtype, default_int_dtype)
    infer_domain_and_extents(program, domain)


def infer_inputs_and_outputs(program: sast.Program):
    """
    Modifies all inputs and outputs of statements and computation blocks in a Stencil IR program.
    Operates in-place.

    :param program: The Stencil IR program to traverse.
    """
    # Programs' inputs and outputs are defined by their arguments
    # Computation block inputs and outputs are defined by their statements and whether the fields are used in
    # future computations.
    # Conditional blocks are the intersection of their respective bodies' inputs and outputs.
    # Statement blocks are defined locally based on their inputs and outputs
    inputs_per_computation: list[set[sast.Identifier]] = []
    outputs_per_computation: list[set[sast.Identifier]] = []

    # Collect statement blocks locally
    for node in helpers.walk(program):
        if isinstance(node, sast.StatementBlock):
            collector = analysis.InputOutputCollector()
            collector.visit(node)
            node.inputs = list(sorted(collector.inputs, key=lambda k: k.name))

    # Collect inputs/outputs per computation and only include globally-necessary fields in a second pass
    for comp in program.computations:
        collector = analysis.InputOutputCollector()
        collector.visit(comp)
        inputs_per_computation.append(collector.inputs)
        outputs_per_computation.append(collector.outputs)

    # Reduce outputs based on subsequent computations
    subsequent_names = set(k.name for k in program.outputs)
    for i, comp in reversed(list(enumerate(program.computations))):
        outputs = outputs_per_computation[i]
        outputs = set(k for k in outputs if k.name in subsequent_names)
        comp.outputs = list(sorted(outputs, key=lambda k: k.name))

        # Figure out intermediate outputs by omission, then remove them from global inputs too
        intermediates = outputs_per_computation[i] - outputs
        comp.inputs = list(sorted(inputs_per_computation[i] - intermediates, key=lambda k: k.name))
        subsequent_names.update(set(k.name for k in inputs_per_computation[i]))


def infer_scalar_types(program: sast.Program, default_float_dtype: sast.ScalarType, default_int_dtype: sast.ScalarType):
    """
    Infers the scalar types of scalars and fields in a Stencil IR program.
    Operates in-place.

    :param program: The Stencil IR program to traverse.
    :param default_float_dtype: The float type to use for float literals (e.g. 0.0) and fields that do not have an
                                explicit type.
    :param default_int_dtype: The integer type to use for integer literals and integral fields that do not have an
                              explicit type.
    """
    field_types: dict[str, sast.ScalarType] = {}  # Stores already inferred types

    # Hierarchy of statements and data types:
    # <Node type>: <Input types> -> <Output types>
    # Program: Fields -> Fields
    #   Computation: Fields -> Fields
    #     Materialize (Identifier): Field -> Field
    #     Statement: Fields -> Field
    #       Assign (Expression): Scalars -> Scalar
    #       Return (Expression): Scalars -> Scalar
    #     IfBlock: Fields -> Fields
    #       Statement (As above)
    pass



def infer_domain_and_extents(program: sast.Program, domain: tuple[int] | None = None):
    """
    Infers the domain size and extents of a Stencil IR program by traversing it backwards.
    Operates in-place.

    :param program: The Stencil IR program to traverse.
    :param domain: An optional 3-tuple representing domain size (x, y, z). If not given, existing domain size will
                   be used or "?" will remain.
    """
    halo = (0, 0, 0)
    field_types: dict[str, sast.FieldType] = {}

    pass


#########################################################################################
# Internal functions


def _result_type_of(*args: sast.ScalarType, optype: str | None = None) -> sast.ScalarType:
    """
    Returns the result scalar type of multiple operands.

    :param args: The arguments.
    :param optype: An optional operation type (e.g., "*") to determine specific behavior.
    :return: The resulting scalar type.
    """
    assert len(args) >= 1
    if optype == 'not':  # Boolean not
        return sast.ScalarType.bool
    if optype in ('>', '>=', '<', '<=', '==', '!='):  # Comparison operators
        return sast.ScalarType.bool

    # Generic upcasting of types
    max_bit_width_float = None, None
    max_bit_width_uint = None, None
    max_bit_width_int = None, None
    for arg in args:
        if arg in (sast.ScalarType.f16, sast.ScalarType.f32, sast.ScalarType.f64):
            # Floating point
            if max_bit_width_float[0] is None or max_bit_width_float[0] < sast.BIT_WIDTH[arg]:
                max_bit_width_float = sast.BIT_WIDTH[arg], arg
        elif arg in (sast.ScalarType.u8, sast.ScalarType.u16, sast.ScalarType.u32):
            # Unsigned integer
            if max_bit_width_uint[0] is None or max_bit_width_uint[0] < sast.BIT_WIDTH[arg]:
                max_bit_width_uint = sast.BIT_WIDTH[arg], arg
        else:
            # Signed integer
            if max_bit_width_int[0] is None or max_bit_width_int[0] < sast.BIT_WIDTH[arg]:
                max_bit_width_int = sast.BIT_WIDTH[arg], arg

    # Specific math calls make the result floating point
    if optype in ('sqrt', 'cbrt'):
        if max_bit_width_float[0] is None:  # If no floating-point value exists, use other arguments
            max_bit_width = max(sast.BIT_WIDTH[arg] for arg in args)
            max_bit_width = max(16, max_bit_width)
            max_bit_width_float = max_bit_width, sast.ScalarType[f'f{max_bit_width}']

    # Evaluate rules as per IR specification
    if max_bit_width_float[0] is not None:
        return max_bit_width_float[1]
    if max_bit_width_uint[0] is not None:
        return max_bit_width_uint[1]
    return max_bit_width_int[1]


def _infer_expression(expr: sast.Expression, field_types: dict[str, sast.ScalarType],
                      default_float_dtype: sast.ScalarType, default_int_dtype: sast.ScalarType) -> sast.ScalarType:
    """
    Type-infers the result of a scalar expression.

    :param expr: The Expression to type-infer.
    :param field_types: A dictionary mapping identifier names to their underlying types.
    :param default_float_type: The default floating point type to use when a literal is encountered.
    :param default_int_type: The default integer type to use when a literal is encountered.
    :return: The resulting scalar type.
    """
    val = expr.value

    # Constant literals
    if isinstance(val, int):
        return default_int_dtype
    if isinstance(val, float):
        return default_float_dtype

    # Fields
    if isinstance(val, sast.Identifier):
        return field_types[val.name]
    if isinstance(val, sast.Subscript):
        return field_types[val.value.name]

    nested_infer_expression = lambda ex: _infer_expression(ex, field_types, default_float_dtype, default_int_dtype)

    # Operators
    if isinstance(val, sast.Expression):
        return nested_infer_expression(val.value)
    if isinstance(val, sast.UnaryOperator):
        return _result_type_of(nested_infer_expression(val.value), optype=val.op)
    if isinstance(val, sast.BinaryOperator):
        return _result_type_of(nested_infer_expression(val.left), nested_infer_expression(val.right), optype=val.op)
    if isinstance(val, sast.TernaryOperator):
        # Determine types by the wider of the two possible values
        return _result_type_of(
            nested_infer_expression(val.true_value), nested_infer_expression(val.false_value), optype=None)
    if isinstance(val, sast.MathCall):
        return _result_type_of(*(nested_infer_expression(arg) for arg in val.arguments), optype=val.func)

    raise TypeError(f'Unidentified AST type {type(val)}')
