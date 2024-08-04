"""
Contains type/extent inference functionality for the Stencil IR.
"""
from spatialstencil.syntax.stencil_ir import astnodes as sast


def infer_types(program: sast.Program,
                default_float_dtype: sast.ScalarType = sast.ScalarType.f32,
                default_int_dtype: sast.ScalarType = sast.ScalarType.i32,
                domain: tuple[int] | None = None,
                halo: tuple[int] | None = None):
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
    :param halo: An optional 3-tuple representing halo extents (x, y, z).
    """
    pass


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


def infer_domain_and_extents(program: sast.Program, halo: tuple[int] | None = None):
    """
    Infers the domain size and extents of a Stencil IR program by traversing it backwards.
    Operates in-place.

    :param program: The Stencil IR program to traverse.
    :param halo: Initial extents to compute for the outputs. If not given, assumes (0, 0, 0).
    """
    halo = halo or (0, 0, 0)
    field_extents: dict[str, list[tuple[int]]] = {}
    # TODO: Materialize stops extent propagation

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
