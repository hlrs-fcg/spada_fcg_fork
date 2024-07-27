import ast
from collections import defaultdict
from spatialstencil.syntax.gt4py import astnodes as gtast
from spatialstencil.syntax.helpers import ASTFindReplace
from spatialstencil.syntax.stencil_ir import astnodes


def lower_gt4py_to_stencil_ir(program: gtast.GTProgram) -> astnodes.Program:
    """
    Takes a GT4Py program (as AST) and returns a logical IR program.
    """

    # Constant propagation
    constant_propagation(program)

    # Unique naming
    field_versioning(program)

    # TODO: Build new tree structure (that matches the language)

    # TODO: Perform type/shape inference in stencil IR language

    # TODO: Insert materialize for all fields
    pass


def field_versioning(program: gtast.GTProgram):
    """
    Ensures every assignment target is unique by setting versions for each field.
    """
    names = set(program.fields)
    name_to_version = defaultdict(int)
    replacements = {}
    for comp in program.computations:
        for intvl in comp.intervals:
            for stmt in intvl.statements:
                # First, replace elements in body (to avoid self-reference clashes)
                stmt.body = ASTFindReplace(replacements).visit(stmt.body)

                # Name clash, add version
                if stmt.target in names:
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
                stmt.body = ASTFindReplace(constants).visit(stmt.body)

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


