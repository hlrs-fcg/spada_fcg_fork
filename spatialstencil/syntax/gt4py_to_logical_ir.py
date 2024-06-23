import ast
from collections import defaultdict
from spatialstencil.syntax import gt4py_parser as gt, astnodes


def lower_gt4py_to_logical_ir(program: gt.GTProgram) -> astnodes.Program:
    """
    Takes a GT4Py program (as AST) and returns a logical IR program.
    """

    # Constant propagation
    constant_propagation(program)

    # Unique naming
    field_versioning(program)

    # TODO: Type/shape inference
    pass


def field_versioning(program: gt.GTProgram):
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


def constant_propagation(program: gt.GTProgram):
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
                elif isinstance(stmt.body, ast.Expr) and isinstance(
                        stmt.body.value, ast.Constant):
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


class ASTFindReplace(ast.NodeTransformer):
    """
    Finds and replaces a name with another value
    """

    def __init__(self, repldict: dict[str, ast.AST]):
        """
        Creates a find-and-replace AST node transformer.

        :param repldict: A dictionary mapping a source name to a target replacement AST node.
        """
        self.replace_count = 0
        self.repldict = repldict
        # If ast.Names were given, use them as keys as well
        self.repldict.update({
            k.id: v
            for k, v in self.repldict.items() if isinstance(k, ast.Name)
        })

    def visit_Name(self, node: ast.Name):
        if node.id in self.repldict:
            val = self.repldict[node.id]
            if isinstance(val, ast.AST):
                new_node = ast.copy_location(val, node)
            else:
                new_node = ast.copy_location(
                    ast.parse(str(self.repldict[node.id])).body[0].value, node)
            self.replace_count += 1
            return new_node

        return self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword):
        if node.arg in self.repldict:
            val = self.repldict[node.arg]
            if isinstance(val, ast.AST):
                val = ast.unparse(val)
            node.arg = val
            self.replace_count += 1
        return self.generic_visit(node)
