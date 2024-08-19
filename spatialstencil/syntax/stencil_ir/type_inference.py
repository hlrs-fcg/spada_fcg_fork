"""
Contains type/extent inference functionality for the Stencil IR.
"""
from spatialstencil.syntax.stencil_ir import irnodes as sast, irnodes
from spatialstencil.syntax.stencil_ir import analysis
from collections import defaultdict
import copy
import math
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
    infer_scalar_types(program, default_float_dtype, default_int_dtype)
    infer_field_extents(program)

    print(program.as_ir())

    infer_field_domains(program, domain)


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
    for node in program.walk():
        if isinstance(node, sast.StatementBlock):
            collector = analysis.InputOutputCollector()
            collector.visit(node)
            node.inputs = _unique_id_list(collector.inputs, False)
            node.operation_type.source = node.operation_type.source[:len(node.inputs)]  # Adjust type information
            node.operation_type.destination = node.operation_type.destination[:len(node.body[-1].values)]

    # Collect inputs/outputs per computation and only include globally-necessary fields in a second pass
    for comp in program.computations:
        collector = analysis.InputOutputCollector()
        collector.visit(comp)
        inputs_per_computation.append(collector.inputs)
        outputs_per_computation.append(collector.outputs)

        # Set the inputs to be anything that is used and defined outside (i.e., not overridden)
        comp.inputs = _unique_id_list(collector.inputs - collector.outputs, False)
        comp.operation_type.source = comp.operation_type.source[:len(comp.inputs)]  # Adjust type information

        # Initialize outputs to final outputs of the block
        comp.outputs = _unique_id_list(collector.outputs, True)

    # Reduce outputs based on usage in subsequent computations
    subsequent_names = set(k.name for k in program.outputs)
    for i, comp in reversed(list(enumerate(program.computations))):
        comp.outputs = [out for out in comp.outputs if out.name in subsequent_names]
        comp.operation_type.destination = comp.operation_type.destination[:len(comp.outputs)]  # Adjust type information
        subsequent_names.update(set(k.name for k in comp.inputs + comp.outputs))


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
    inferrer = TypeInference(default_float_dtype, default_int_dtype)

    # Collect program global fields. If type is unknown, use default float type
    for field, dtype in zip(program.inputs, program.operation_type.source):
        if dtype.dtype == sast.ScalarType.UNKNOWN:
            dtype.dtype = default_float_dtype
        inferrer.field_types[field.name] = dtype.dtype
    for field, dtype in zip(program.outputs, program.operation_type.destination):
        if dtype.dtype == sast.ScalarType.UNKNOWN:
            dtype.dtype = default_float_dtype
        inferrer.field_types[field.name] = dtype.dtype

    # Run type inference throughout program
    inferrer.visit(program)


class TypeInference(sast.NodeTransformer):

    def __init__(self, default_float_dtype: sast.ScalarType, default_int_dtype: sast.ScalarType):
        super().__init__()
        self.field_types: dict[str, sast.ScalarType] = {}  # Types that were already inferred
        self.float_dtype = default_float_dtype
        self.int_dtype = default_int_dtype

    def _modify_typeinfo(self, operation_type: sast.OperationType, inputs: list[sast.Identifier],
                         outputs: list[sast.Identifier]):
        """
        Helper function that updates the type information based on inferred types.
        """
        for i, (name, src) in enumerate(zip(inputs, operation_type.source)):
            scalartype = self.field_types[name.name]
            if isinstance(src, sast.FieldType):
                src.dtype = scalartype
            else:  # Scalar type
                operation_type.source[i] = scalartype

        if operation_type.destination:
            for i, (name, dst) in enumerate(zip(outputs, operation_type.destination)):
                scalartype = self.field_types[name.name]
                if isinstance(dst, sast.FieldType):
                    dst.dtype = scalartype
                else:  # Scalar type
                    operation_type.destination[i] = scalartype

    # Scalar operations
    def visit_ReturnOp(self, node: sast.ReturnOp):
        for i, val in enumerate(node.values):
            node.operation_type.source[i] = _infer_expression(val, self.field_types, self.float_dtype, self.int_dtype)
        return node

    def visit_AssignOp(self, node: sast.AssignOp):
        output_type = _infer_expression(node.value, self.field_types, self.float_dtype, self.int_dtype)
        node.operation_type.source[0] = output_type
        if node.result.name not in self.field_types:
            self.field_types[node.result.name] = output_type
        node.operation_type.destination[0] = output_type
        return node

    # Field operations
    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        assert node.value.name in self.field_types
        node.operation_type.source[0].dtype = self.field_types[node.value.name]
        node.operation_type.destination[0].dtype = self.field_types[node.value.name]
        self.field_types[node.result.name] = self.field_types[node.value.name]
        return node

    # Non-leaf blocks
    def visit_StatementBlock(self, node: sast.StatementBlock):
        # First traverse children
        node = self.generic_visit(node)

        # Input types should aleady exist
        for src, src_type in zip(node.inputs, node.operation_type.source):
            src_type.dtype = self.field_types[src.name]

        # Use return value to infer output types
        assert isinstance(node.body[-1], sast.ReturnOp)
        retvals = node.body[-1].operation_type.source
        for dst, retval, dst_type in zip(node.outputs, retvals, node.operation_type.destination):
            dst_type.dtype = retval
            self.field_types[dst.name] = retval

        return node

    def visit_IfBlock(self, node: sast.IfBlock):
        # First traverse children
        node = self.generic_visit(node)

        # Input type should aleady exist
        # TODO(later): Verify that condition / return types match across branches in a separate validation pass
        node.operation_type.source[0].dtype = self.field_types[node.condition.name]

        # Use return value to infer output types
        assert isinstance(node.body[-1], sast.ReturnOp)
        retvals = node.body[-1].operation_type.source
        for dst, retval, dst_type in zip(node.outputs, retvals, node.operation_type.destination):
            dst_type.dtype = retval
            self.field_types[dst.name] = retval

        return node

    def visit_ComputationBlock(self, node: sast.ComputationBlock):
        node = self.generic_visit(node)
        self._modify_typeinfo(node.operation_type, node.inputs, node.outputs)
        return node


def infer_field_extents(program: sast.Program):
    """
    Infers the extents of a Stencil IR program by traversing it twice.
    Operates in-place.

    :param program: The Stencil IR program to traverse.
    """
    field_extents: dict[str, dict[tuple[int | None], set[tuple[int | None]]]] = {}

    # Start with outputs. Extents always start at (0, 0, 0)
    assert isinstance(program.operation_type.destination, list)
    for field, dtype in zip(program.outputs, program.operation_type.destination):
        if dtype.extent.is_unknown():
            dtype.extent.extents = [sast.OffsetAndInterval((0, 0, 0))]
        field_extents[field.name] = defaultdict(set)
        for oi in dtype.extent.extents:
            field_extents[field.name][oi.interval].add(oi.values)

    # Visit entire program and collect extents
    field_extents.update(analysis.collect_extents(program))

    # Transform the program by assigning extents to field types
    ExtentAssigner(field_extents).visit(program)


def infer_field_domains(program: sast.Program, domain: sast.Cartesian | None = None):
    """
    Infers the domain size of a Stencil IR program by traversing it backwards.
    Operates in-place.

    :param program: The Stencil IR program to traverse.
    :param domain: An optional 3-tuple representing domain size (x, y, z). If not given, existing domain size will
                   be used or "?" will remain.
    """
    if domain is None:  # Nothing to do
        return
    field_domains: dict[str, sast.Cartesian] = {}

    # Start with outputs. Use halo for extents.
    assert isinstance(program.operation_type.destination, list)
    for field, dtype in zip(program.outputs, program.operation_type.destination):
        if dtype.domain.is_unknown():
            dtype.domain = domain
        field_domains[field.name] = dtype.domain

    for field, dtype in zip(program.inputs, program.operation_type.source):
        field_domains[field.name] = _infer_domain_from_extents(domain, dtype.extent)
        print(field.name, field_domains[field.name] )

    # Gather failed identifiers for warnings
    potentially_unknown_identifiers: set[str] = set()

    # Propagate backwards through statements from end of program
    for node in reversed(list(program.walk())):
        if isinstance(node, sast.StatementBlock):
            # Gather the local domain
            stmt_domain = None
            for out, outtype in zip(node.outputs, node.operation_type.destination):
                if stmt_domain is None:
                    if not outtype.domain.is_unknown():
                        stmt_domain = outtype.domain
                    elif out.name in field_domains:
                        stmt_domain = field_domains[out.name]
                else:
                    if not outtype.domain.is_unknown() and outtype.domain != stmt_domain:
                        raise ValueError('Ambiguous domains found when processing multiple '
                                         f'statement outputs: {stmt_domain} != {outtype.domain}')
                    if out.name in field_domains and field_domains[out.name] != stmt_domain:
                        raise ValueError('Ambiguous domains found when processing multiple '
                                         f'statement outputs: {stmt_domain} != {field_domains[out.name]}')
            if stmt_domain is None:
                for out in node.outputs:
                    potentially_unknown_identifiers.add(out.name)
                continue

            # Compute input domains based on extents and output
            for inp, inptype in zip(node.inputs, node.operation_type.source):
                new_domain = _infer_domain_from_extents(stmt_domain, inptype.extent)
                # Take max value from current domain if in dictionary
                if inp.name in field_domains:
                    dom = field_domains[inp.name]
                    field_domains[inp.name] = dom.union(new_domain)
                else:
                    # Else, copy
                    field_domains[inp.name] = copy.deepcopy(new_domain)
        elif isinstance(node, sast.MaterializeOp):
            if node.result.name not in field_domains:
                warnings.warn(f'Cannot infer domain size from materialization of "%{node.value.name}"')
                continue
            field_domains[node.value.name] = field_domains[node.result.name]

    # Warn on still-unknown identifiers
    for identifier in potentially_unknown_identifiers:
        if identifier not in field_domains:
            warnings.warn(f'Could not infer domain size for "%{identifier}"')

    # Assign inferred domain sizes across Stencil IR program
    DomainAssigner(field_domains).visit(program)


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
    if optype in ('not', 'and', 'or'):  # Boolean operators
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
        elif arg in (sast.ScalarType.bool, sast.ScalarType.u8, sast.ScalarType.u16, sast.ScalarType.u32):
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
        if max_bit_width_int[0] is not None and max_bit_width_int[0] > max_bit_width_uint[0]:
            return max_bit_width_int[1]
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


def _infer_domain_from_extents(output_domain: sast.Cartesian, extents: sast.Extent) -> irnodes.Cartesian:
    """
    Given the output domain and extents, infers the domain size of the input field.
    Assuming that the output is accessed at offset (0, 0, 0).
    :param output_domain:
    :param extents:
    :return:
    """
    #print(output_domain, extents)
    current_domain = copy.deepcopy(output_domain)

    for e in extents.extents:
        # Convert the extent interval into a cartesian domain representing the output
        extent_domain = output_domain.intersect_with_ranges(e.interval)
        # Expand the domain with the extent values
        extent_domain = extent_domain.add(e.values)

        current_domain = current_domain.union(extent_domain)
        #print(current_domain)

    return current_domain


def _unique_id_list(identifiers: set[sast.Identifier], latest_version: bool) -> list[sast.Identifier]:
    """
    Makes a list of uniquely-named identifiers from a set thereof. The ordering is deterministic (sorted)
    and can use either the earliest or latest version.

    :param identifiers: Set of identifiers.
    :param latest_version: If True, keeps the latest identifier version. Otherwise, uses the earliest one.
    """
    names = set()
    versions = {}
    func = max if latest_version else min
    for k in identifiers:
        names.add(k.name)
        if k.name not in versions:
            versions[k.name] = k.version
        else:
            versions[k.name] = func(k.version, versions[k.name])

    return [sast.Identifier(name, versions[name]) for name in sorted(names)]


def sort_extents(extent_set: dict[tuple[int | None], set[tuple[int | None]]]):
    """
    Yields a flat list of sorted extents as it should appear in a canonical Stencil IR.
    Substitutes None entries for infinity.
    """
    # None means infinity in the dictionary keys
    newdict = {tuple(math.inf if kk is None else kk for kk in k): v for k, v in extent_set.items()}
    for k, tuples in sorted(newdict.items()):
        oldkey = tuple(None if kk == math.inf else kk for kk in k)  # Recover old values

        # None means "?" in the dictionary values, which must be last as well
        newtuples = [tuple(math.inf if t is None else t for t in tup) for tup in tuples]
        for tup in sorted(newtuples):
            oldtup = tuple(None if kk == math.inf else kk for kk in tup)
            yield oldkey, oldtup


class ExtentAssigner(sast.NodeTransformer):
    """
    Sets extents based on given dictionary.
    """

    def __init__(self, field_extents: dict[str, dict[tuple[int | None], set[tuple[int | None]]]]):
        super().__init__()
        self.field_extents = field_extents

    def _modify_typeinfo(self, operation_type: sast.OperationType, inputs: list[sast.Identifier],
                         outputs: list[sast.Identifier]):
        """
        Helper function that updates the type information based on inferred types.
        """
        for name, src in zip(inputs, operation_type.source):
            if name.name in self.field_extents:
                extent_set = self.field_extents[name.name]
                if isinstance(src, sast.FieldType):
                    src.extent.extents = [
                        sast.OffsetAndInterval(ex, interval) for interval, ex in sort_extents(extent_set)
                    ]

        if operation_type.destination:
            for name, dst in zip(outputs, operation_type.destination):
                if name.name in self.field_extents:
                    extent_set = self.field_extents[name.name]
                    if isinstance(dst, sast.FieldType):
                        dst.extent.extents = [
                            sast.OffsetAndInterval(ex, interval) for interval, ex in sort_extents(extent_set)
                        ]

    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        self._modify_typeinfo(node.operation_type, [node.value], [node.result])
        return self.generic_visit(node)

    def visit_StatementBlock(self, node: sast.StatementBlock):
        self._modify_typeinfo(node.operation_type, node.inputs, node.outputs)
        return self.generic_visit(node)

    def visit_IfBlock(self, node: sast.IfBlock):
        self._modify_typeinfo(node.operation_type, [node.condition], node.outputs)
        return self.generic_visit(node)

    def visit_ComputationBlock(self, node: sast.ComputationBlock):
        self._modify_typeinfo(node.operation_type, node.inputs, node.outputs)
        return self.generic_visit(node)

    def visit_Program(self, node: sast.Program):
        self._modify_typeinfo(node.operation_type, node.inputs, node.outputs)
        return self.generic_visit(node)


class DomainAssigner(sast.NodeTransformer):
    """
    Sets domain based on given dictionary.
    """

    def __init__(self, field_domains: dict[str, sast.Cartesian]):
        super().__init__()
        self.field_domains = field_domains

    def _modify_typeinfo(self, operation_type: sast.OperationType, inputs: list[sast.Identifier],
                         outputs: list[sast.Identifier]):
        """
        Helper function that updates the type information based on inferred types.
        """
        for name, src in zip(inputs, operation_type.source):
            if name.name in self.field_domains and isinstance(src, sast.FieldType) and src.domain.is_unknown():
                src.domain = copy.deepcopy(self.field_domains[name.name])

        if operation_type.destination:
            for name, dst in zip(outputs, operation_type.destination):
                if name.name in self.field_domains and isinstance(dst, sast.FieldType) and dst.domain.is_unknown():
                    dst.domain = copy.deepcopy(self.field_domains[name.name])

    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        self._modify_typeinfo(node.operation_type, [node.value], [node.result])
        return self.generic_visit(node)

    def visit_StatementBlock(self, node: sast.StatementBlock):
        self._modify_typeinfo(node.operation_type, node.inputs, node.outputs)
        return self.generic_visit(node)

    def visit_IfBlock(self, node: sast.IfBlock):
        self._modify_typeinfo(node.operation_type, [node.condition], node.outputs)
        return self.generic_visit(node)

    def visit_ComputationBlock(self, node: sast.ComputationBlock):
        self._modify_typeinfo(node.operation_type, node.inputs, node.outputs)
        return self.generic_visit(node)

    def visit_Program(self, node: sast.Program):
        self._modify_typeinfo(node.operation_type, node.inputs, node.outputs)
        return self.generic_visit(node)
