import ast
from collections import defaultdict
from spatialstencil.syntax.gt4py import astnodes as gtast
from spatialstencil.syntax import helpers
from spatialstencil.syntax.stencil_ir import irnodes as sast, type_inference


def lower_gt4py_to_stencil_ir(program: gtast.GTProgram,
                              default_float_dtype: sast.ScalarType = sast.ScalarType.f32,
                              default_int_dtype: sast.ScalarType = sast.ScalarType.i32,
                              domain: tuple[int] | None = None,
                              materialize: bool = True) -> sast.Program:
    """
    Takes a GT4Py program (as AST) and returns a logical IR program.

    :param program: The GT4Py AST to lower.
    :param default_float_dtype: The float type to use for float literals (e.g. 0.0) and fields that do not have an
                                explicit type.
    :param default_int_dtype: The integer type to use for integer literals and integral fields that do not have an
                              explicit type.
    :param domain: An optional domain size to compute the stencil on. If not given, keeps shapes unknown for
                   future shape inference.
    :param materialize: If True, runs a pass that materializes all intermediate values.
    :return: A Stencil IR node representing the lowered program.
    """

    # Constant propagation
    constant_propagation(program)

    # Unique naming
    field_versioning(program)

    # Build new tree structure (that matches the language)
    new_ast = convert_gt4py_ast_to_stencil_ast(program, default_float_dtype, default_int_dtype)

    # Infer which fields are intermediate (before materialize pass)
    type_inference.infer_inputs_and_outputs(new_ast)

    # Insert materialize for all fields
    if materialize:
        # Perform a first round of shape inference to skip certain materializations
        type_inference.infer_field_extents(new_ast)

        new_ast = MaterializeIntermediates().visit(new_ast)

    # Perform type/shape inference in stencil IR language
    type_inference.infer_types(new_ast, default_float_dtype, default_int_dtype, domain)

    return new_ast


def field_versioning(program: gtast.GTProgram):
    """
    Ensures every assignment target is unique by setting versions for each field.
    """
    names = set(program.fields)
    name_to_version = defaultdict(int)
    used_identifiers: set[str] = set()
    replacements = {}
    for comp in program.computations:
        for intvl in comp.intervals:
            for stmt in intvl.statements:
                if not isinstance(stmt, gtast.GTComputeStatement):
                    continue

                # First, replace elements in body (to avoid self-reference clashes)
                replacer = helpers.ASTFindReplace(replacements)
                stmt.body = replacer.visit(stmt.body)
                used_identifiers.update(replacer.encountered_names)

                # Name clash, add version
                if stmt.target in names:
                    # Special case: if the target is an output is never read/overwritten, keep version zero
                    if (stmt.target in program.fields and stmt.target not in name_to_version and
                            stmt.target not in used_identifiers):
                        name_to_version[stmt.target] = 0
                        continue

                    # TODO(later): Do not make new version if intervals do not overlap?
                    old_name = stmt.target
                    name_to_version[stmt.target] += 1
                    stmt.target = f'{stmt.target}#{name_to_version[stmt.target]}'
                    replacements[old_name] = ast.Name(id=stmt.target)
                else:
                    names.add(stmt.target)


def constant_propagation(program: gtast.GTProgram):
    """
    Replaces all subsequent appearances of a constant with its value.
    """
    for comp in program.computations:
        for intvl in comp.intervals:
            constants = {}
            statements_to_remove = []
            for i, stmt in enumerate(intvl.statements):
                if not isinstance(stmt, gtast.GTComputeStatement):
                    continue

                # If the statement was overwritten with a non-constant, remove
                if stmt.target in constants:
                    del constants[stmt.target]

                # Find out if this is a constant
                if isinstance(stmt.body, ast.Constant):
                    constants[stmt.target] = stmt.body.value
                    statements_to_remove.append(i)
                    continue
                elif isinstance(stmt.body, ast.Expr) and isinstance(stmt.body.value, ast.Constant):
                    constants[stmt.target] = stmt.body.value.value
                    statements_to_remove.append(i)
                    continue

                # Find constants within expression and replace
                stmt.body = helpers.ASTFindReplace(constants).visit(stmt.body)

                # If this statement is now a constant, make it so
                try:
                    val = eval(ast.unparse(stmt.body))
                    constants[stmt.target] = val
                    statements_to_remove.append(i)
                    continue
                except:
                    pass

            # After looping over statements, remove constants
            for i in reversed(statements_to_remove):
                intvl.statements.pop(i)


class MaterializeIntermediates(sast.NodeTransformer):

    def __init__(self):
        super().__init__()
        self.do_not_materialize: set[str] = set()
        self.name_translation: dict[str, tuple[str, int]] = {}
        self.allnames: set[str] = set()

    def _find_new_name(self, name: str):
        """
        Finds a new name that is not used by the rest of the program.
        """
        new_name = f'{name}_mat'
        i = 1
        while new_name in self.allnames:
            new_name = f'{name}_mat{i}'
            i += 1
        self.allnames.add(new_name)
        return new_name

    def visit_Program(self, node: sast.Program):
        # Input/output fields are always materialized
        self.do_not_materialize |= set(n.name for n in node.inputs)
        self.do_not_materialize |= set(n.name for n in node.outputs)

        # Collect all used identifier names (to assign a new name)
        for subnode in helpers.walk(node):
            if isinstance(subnode, sast.Identifier):
                self.allnames.add(subnode.name)

        return self.generic_visit(node)

    def visit_Identifier(self, node: sast.Identifier):
        cur_node = node
        while cur_node.name in self.name_translation:
            new_name, version_diff = self.name_translation[cur_node.name]
            cur_node = sast.Identifier(name=new_name, version=cur_node.version - version_diff)
        if cur_node is not node:
            return cur_node
        return self.generic_visit(node)

    def visit_ComputationBlock(self, node: sast.ComputationBlock):
        for i, inp in enumerate(node.inputs):
            node.inputs[i] = self.visit(inp)

        new_body = []
        for stmt in node.body:
            # Add statement to new body
            stmt = self.visit(stmt)
            new_body.append(stmt)

            if isinstance(stmt, sast.MaterializeOp):
                continue  # No need to rematerialize if applied more than once

            # Try to find results
            results: list[sast.Identifier] | None = None
            stmt: sast.StatementBlock | sast.IfBlock
            results = stmt.outputs
            result_typeinfo = stmt.typeinfo.destination

            # Append materialize after statement
            if results:
                if result_typeinfo:
                    # Do not materialize if horizontal offsets (i.e., [0:1]) are all zero
                    results = [
                        r for r, typeinfo in zip(results, result_typeinfo)
                        if any(v != 0 for e in typeinfo.extent.extents for v in e.values[0:1])
                    ]

                results = [r for r in results if r.name not in self.do_not_materialize]

                for result in results:
                    # Find a new name and version difference to assign
                    materialized = sast.Identifier(self._find_new_name(result.name), version=0)
                    self.name_translation[result.name] = (materialized.name, result.version)
                    new_body.append(
                        sast.MaterializeOp(
                            result=materialized,
                            value=result,
                            typeinfo=sast.TypeInfo([sast.FieldType.empty()], [sast.FieldType.empty()])))

        for i, out in enumerate(node.outputs):
            node.outputs[i] = self.visit(out)

        return sast.ComputationBlock(node.outputs, node.inputs, node.schedule, node.interval, node.typeinfo, new_body)


def convert_gt4py_ast_to_stencil_ast(program: gtast.GTProgram, default_float_dtype: sast.ScalarType,
                                     default_int_dtype: sast.ScalarType) -> sast.Program:
    input_fields: set[str] = set()
    output_fields: set[str] = set()
    computations: list[sast.ComputationBlock] = []
    field_type_by_name = {
        k: _gt4py_to_stencil_ir_type(v, default_float_dtype, default_int_dtype)
        for k, v in zip(program.fields, program.field_types)
    }
    fields = set(program.fields)

    for computation in program.computations:
        for interval in computation.intervals:  # Computation + interval becomes an spst.computation block
            # Horizontal intervals always cover the entire field in GT4Py
            xintvl = sast.Interval(0, None)
            yintvl = sast.Interval(0, None)
            zintvl = sast.Interval(interval.start, interval.end)

            cbody, cinputs, coutputs = _convert_interval_to_computation_body(interval.statements)

            _aggregate_fields(input_fields, cinputs, fields)
            _aggregate_fields(output_fields, coutputs, fields)

            # Types will be refined by type inference later
            computations.append(
                sast.ComputationBlock(
                    coutputs, cinputs, sast.ComputationType[computation.computation_type.name],
                    (xintvl, yintvl, zintvl),
                    sast.TypeInfo([sast.FieldType.empty() for _ in cinputs],
                                  [sast.FieldType.empty() for _ in coutputs]), cbody))

    return sast.Program(
        outputs=[sast.Identifier(field) for field in sorted(output_fields)],
        name=program.name,
        inputs=[sast.Identifier(field) for field in sorted(input_fields)],
        attributes={},
        typeinfo=sast.TypeInfo([field_type_by_name[field] for field in sorted(input_fields)],
                               [field_type_by_name[field] for field in sorted(output_fields)]),
        computations=computations,
    )


# Helper functions
def _aggregate_fields(aggto: set[str], aggfrom: set[sast.Identifier], argnames: set[str]):
    """
    Collect fields that appear in ``argnames`` from ``aggfrom`` and add them to ``aggto``.
    """
    aggto |= set(f.name for f in aggfrom) & argnames


def _parse_field(name: str) -> sast.Identifier:
    """
    Parses a versioned field by name.
    """
    if '#' not in name:
        return sast.Identifier(name)
    fname, version = name.split('#')
    return sast.Identifier(fname, int(version))


def _gt4py_to_stencil_ir_type(dtype: gtast.FieldType, default_float_dtype: sast.ScalarType,
                              default_int_dtype: sast.ScalarType) -> sast.FieldType:
    result = sast.FieldType.empty()

    # TODO(later): Try to use GT4Py type annotations if explicit
    if dtype == gtast.FieldType.Field3D:
        result.dtype = default_float_dtype
    elif dtype == gtast.FieldType.FieldIJ:
        result.dtype = default_float_dtype
        result.domain = sast.Cartesian(None, None, 1)
    elif dtype == gtast.FieldType.FieldI:
        result.dtype = default_float_dtype
        result.domain = sast.Cartesian(None, 1, 1)
    elif dtype == gtast.FieldType.FieldJ:
        result.dtype = default_float_dtype
        result.domain = sast.Cartesian(1, None, 1)
    elif dtype == gtast.FieldType.FieldK:
        result.dtype = default_float_dtype
        result.domain = sast.Cartesian(1, 1, None)
    elif dtype == gtast.FieldType.int:
        result.dtype = default_int_dtype
        result.domain = sast.Cartesian(1, 1, 1)
        result.extent = sast.Extent([sast.OffsetAndInterval((0, 0, 0))])
    elif dtype == gtast.FieldType.float:
        result.dtype = default_float_dtype
        result.domain = sast.Cartesian(1, 1, 1)
        result.extent = sast.Extent([sast.OffsetAndInterval((0, 0, 0))])
    else:
        raise TypeError(f'Unsupported field type "{dtype}"')

    return result


def _convert_interval_to_computation_body(
    statements: list[gtast.GTStatement],
    in_conditional: bool = False,
) -> tuple[list[sast.StatementBlock | sast.IfBlock], set[sast.Identifier], set[sast.Identifier]]:
    """
    Converts the body of a GT4Py interval to a Stencil IR computation block
    """
    body = []
    inputs, outputs = set(), set()

    for stmt in statements:
        if isinstance(stmt, gtast.GTComputeStatement):
            # Collect I/O information
            stmt_inputs = []
            ignore = set()
            outputs.add(_parse_field(stmt.target))
            for node in ast.walk(stmt.body):
                if node in ignore:
                    continue
                if isinstance(node, ast.Call):
                    ignore.add(node.func)
                if isinstance(node, ast.Name):
                    identifier = _parse_field(node.id)
                    inputs.add(identifier)
                    if identifier not in stmt_inputs:
                        stmt_inputs.append(identifier)

            # Construct statement
            body.append(
                sast.StatementBlock(
                    outputs=[_parse_field(stmt.target)],
                    inputs=stmt_inputs,
                    attributes=[],
                    typeinfo=sast.TypeInfo([sast.FieldType.empty() for _ in stmt_inputs], [sast.FieldType.empty()]),
                    body=[
                        sast.ReturnOp([sast.Expression(OperationConverter().visit(stmt.body))]),
                    ]))

        elif isinstance(stmt, gtast.GTIfStatement):
            if in_conditional:
                # Assert no nested conditionals
                raise SyntaxError('Nested conditionals are not allowed in Stencil IR')
            assert isinstance(stmt.condition, ast.Name), 'Conditional must only apply on a mask field'

            # Calls the computation body converter recursively, collecting the outputs and adding return values
            all_stmt_outputs: set[sast.Identifier]

            # if block
            stmt_body, _, all_stmt_outputs = _convert_interval_to_computation_body(stmt.body, True)

            # elif blocks
            else_ifs = None
            if stmt.else_ifs:
                else_ifs: list[tuple[sast.Identifier, list[sast.StatementBlock]]] = []
                for elif_cond, elif_body in stmt.else_ifs:
                    elif_body, _, elif_outputs = _convert_interval_to_computation_body(elif_body, True)
                    assert isinstance(elif_cond, ast.Name), 'Else-if conditional must only apply on a mask field'
                    else_ifs.append((_parse_field(elif_cond.id, elif_body)))
                    # Intersect outputs
                    all_stmt_outputs &= elif_outputs

            # else block
            orelse = None
            if stmt.orelse:
                orelse, _, else_outputs = _convert_interval_to_computation_body(stmt.orelse, True)
                all_stmt_outputs &= else_outputs

            # Add overall return values to each branch
            stmt_body.append(
                sast.ReturnOp([sast.Expression(so) for so in sorted(all_stmt_outputs)],
                              [sast.FieldType.empty() for _ in outputs]))
            if else_ifs:
                for _, elif_body in else_ifs:
                    elif_body.append(
                        sast.ReturnOp([sast.Expression(so) for so in sorted(all_stmt_outputs)],
                                      [sast.FieldType.empty() for _ in outputs]))
            if orelse:
                orelse.append(
                    sast.ReturnOp([sast.Expression(so) for so in sorted(all_stmt_outputs)],
                                  [sast.FieldType.empty() for _ in outputs]))

            # Create IR node
            body.append(
                sast.IfBlock(
                    result=list(sorted(all_stmt_outputs)),
                    condition=_parse_field(stmt.condition.id),
                    body=body,
                    else_ifs=else_ifs,
                    orelse=orelse,
                    typeinfo=sast.TypeInfo([sast.FieldType.empty()],
                                           [sast.FieldType.empty() for _ in all_stmt_outputs]),
                ))
        else:
            raise TypeError(f'Unsupported statement type "{type(stmt)}"')

    return body, inputs, outputs


class OperationConverter(ast.NodeTransformer):
    """
    Converts GT4Py operations into Stencil IR operator nodes.
    """
    _UNOPS = {ast.Invert: '~', ast.Not: 'not', ast.UAdd: '+', ast.USub: '-'}

    _BINOPS = {
        ast.Add: '+',
        ast.BitAnd: '&',
        ast.BitOr: '|',
        ast.BitXor: '^',
        ast.Div: '/',
        # ast.FloorDiv: '//',
        ast.LShift: '<<',
        ast.Mod: '%',
        ast.Mult: '*',
        # ast.MatMult: '@',
        ast.Pow: '**',
        ast.RShift: '>>',
        ast.Sub: '-'
    }

    _BOOLOPS = {ast.And: '&&', ast.Or: '||'}

    _CMPOPS = {
        ast.Eq: '==',
        ast.Gt: '>',
        ast.GtE: '>=',
        #ast.In: 'in',
        #ast.NotIn: 'not in'
        #ast.Is: 'is',
        #ast.IsNot: 'is not',
        ast.Lt: '<',
        ast.LtE: '<=',
        ast.NotEq: '!=',
    }

    def visit_UnaryOp(self, node: ast.UnaryOp) -> sast.UnaryOperator:
        return sast.UnaryOperator(
            op=self._UNOPS[type(node.op)],
            value=sast.Expression(self.visit(node.operand)),
        )

    def visit_BinOp(self, node: ast.BinOp) -> sast.BinaryOperator:
        return sast.BinaryOperator(
            left=sast.Expression(self.visit(node.left)),
            op=self._BINOPS[type(node.op)],
            right=sast.Expression(self.visit(node.right)),
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> sast.BinaryOperator:
        # Break down boolean operator to constituent binary operators, processing left to right
        last = self.visit(node.values[0])
        for i in range(1, len(node.values)):
            last = sast.BinaryOperator(
                left=sast.Expression(last),
                op=self._BOOLOPS[type(node.op)],
                right=sast.Expression(self.visit(node.values[i])),
            )
        return last

    def visit_Compare(self, node: ast.Compare) -> sast.BinaryOperator:
        assert len(node.ops) == 1
        assert len(node.comparators) == 1
        return sast.BinaryOperator(
            left=sast.Expression(self.visit(node.left)),
            op=self._CMPOPS[type(node.ops[0])],
            right=sast.Expression(self.visit(node.comparators[0])),
        )

    def visit_IfExp(self, node: ast.IfExp) -> sast.TernaryOperator:
        return sast.TernaryOperator(
            true_value=sast.Expression(self.visit(node.body)),
            test=sast.Expression(self.visit(node.test)),
            false_value=sast.Expression(self.visit(node.orelse)),
        )

    def visit_Constant(self, node: ast.Constant) -> int | float:
        return node.value

    def visit_Subscript(self, node: ast.Subscript) -> sast.Subscript:
        value = self.visit(node.value)
        assert isinstance(node.slice, ast.Tuple)  # Ensures single value rather than slices
        subscript = ast.literal_eval(node.slice)
        return sast.Subscript(value, subscript)

    def visit_Name(self, node: ast.Name) -> sast.Identifier:
        return _parse_field(node.id)

    def visit_Call(self, node: ast.Call) -> None:
        assert isinstance(node.func, ast.Name)
        return sast.MathCall(func=node.func.id, arguments=[sast.Expression(self.visit(arg)) for arg in node.args])


if __name__ == '__main__':
    import sys
    from spatialstencil.syntax.gt4py import parser

    if len(sys.argv) not in (2, 3):
        print('USAGE: python -m spatialstencil.lowering.gt4py_to_stencil_ir <PYTHON FILE> [FUNCTION NAME]')
        exit(1)

    out = parser.parse_file(sys.argv[1])
    if len(sys.argv) == 3:
        out = out[sys.argv[2]]
        new_ast = lower_gt4py_to_stencil_ir(out)
        print(new_ast.as_ir())
    else:
        for fname, func in out.items():
            print('\n====================================')
            print('Function', fname)
            new_ast = lower_gt4py_to_stencil_ir(func, domain=(128, 128, 80))
            print(new_ast.as_ir(), flush=True)
