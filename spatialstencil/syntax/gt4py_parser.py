import ast
import enum
from dataclasses import dataclass
import sys
from typing import TextIO

class ComputationType(enum.Enum):
    PARALLEL = 0
    FORWARD = 1
    BACKWARD = 2

class GTree:
    def pretty(self) -> str:
        """
        A pretty-printed version of the GT4Py stencil tree
        """
        return str(self)

@dataclass
class GTStatement(GTree):
    target: str
    body: ast.Expr

    def pretty(self) -> str:
        return f'      {self.target} = {ast.unparse(self.body)}'

@dataclass
class GTInterval(GTree):
    start: int
    end: int | None
    statements: list[GTStatement]

    def pretty(self) -> str:
        newline = '\n'
        end = 'END' if self.end is None else self.end
        return f'    interval [{self.start}:{end}]:\n{newline.join(i.pretty() for i in self.statements)}'


@dataclass
class GTComputation(GTree):
    computation_type: ComputationType
    intervals: list[GTInterval]

    def pretty(self) -> str:
        newline = '\n'
        return f'  {self.computation_type.name.lower()}:\n{newline.join(i.pretty() for i in self.intervals)}'

@dataclass
class GTProgram(GTree):
    name: str
    fields: list[str]
    computations: list[GTComputation]
    
    def pretty(self) -> str:
        newline = '\n'
        return f'program {self.name} ({", ".join(self.fields)}):\n{newline.join(c.pretty() for c in self.computations)}'

class GTVisitor(ast.NodeVisitor):
    """
    Recursively visits a (valid) GT4Py Python AST and produces the stencil program tree
    """
    def visit_FunctionDef(self, node: ast.FunctionDef) -> GTProgram:
        fields = [arg.arg for arg in node.args.args]
        computations = []
        for subnode in node.body:
            if isinstance(subnode, ast.With):
                computations.append(self.visit_With(subnode))
        return GTProgram(node.name, fields, computations)

    def visit_With(self, node: ast.With):
        assert isinstance(node.items[0].context_expr, ast.Call)
        assert 'computation' in ast.unparse(node.items[0].context_expr.func)
        comptype = node.items[0].context_expr.args
        assert len(comptype) == 1 and isinstance(comptype[0], ast.Name)
        ctype = ComputationType[comptype[0].id]

        if len(node.items) == 1:  # Computation only
            intervals = []
            for stmt in node.body:
                assert isinstance(stmt, ast.With)
                assert len(stmt.items) == 1
                intervals.append(self.visit_interval(stmt))
        elif len(node.items) == 2: # Computation and interval
            intervals = [self.visit_interval(node, 1)]
        else:
            raise SyntaxError('Unexpected number of with items')

        return GTComputation(ctype, intervals)

    def visit_interval(self, node: ast.With, index: int = 0):
        assert isinstance(node.items[index].context_expr, ast.Call)
        interval = node.items[index].context_expr.args
        if len(interval) == 1 and ast.unparse(interval[0]) == '...':
            int_s, int_e = (0, None)
        elif len(interval) == 2:
            int_s, int_e = (ast.literal_eval(interval[0]), ast.literal_eval(interval[1]))
        else:
            raise SyntaxError('Unexpected interval')
        
        stmts = []
        for stmt in node.body:
            assert isinstance(stmt, ast.Assign)
            assert len(stmt.targets) == 1 # Do not allow ``a = b = ...``
            stmts.append(GTStatement(target=ast.unparse(stmt.targets[0]),
                         body=stmt.value))
        return GTInterval(int_s, int_e, stmts)


def parse_function(func: ast.FunctionDef) -> GTProgram:
    return GTVisitor().visit(func)

def parse_string(code: str) -> dict[str, GTree]:
    """
    Parses a string representing a spatial stencil program, returning the
    top-level program AST node.
    
    :param code: A code string in spatial stencil format.
    :return: A Program node representing the root of the AST.
    """
    module = ast.parse(code)
    result = {}

    for stmt in module.body:
        if isinstance(stmt, ast.FunctionDef):
            result[stmt.name] = parse_function(stmt)

    return result

def parse_file(file_or_filename: TextIO | str) -> dict[str, ast.FunctionDef]:
    """
    Parses a file representing a spatial stencil program, returning the
    top-level program AST node.
    
    :param file_or_filename: A file path or handle to an open file to read.
    :return: A Program node representing the root of the AST.
    """
    if isinstance(file_or_filename, str):
        with open(file_or_filename, 'r') as fp:
            return parse_string(fp.read())
    return parse_string(file_or_filename.read())


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        print('USAGE: python -m spatialstencil.syntax.gt4py_parser <PYTHON FILE> [FUNCTION NAME]')
        exit(1)

    out = parse_file(sys.argv[1])
    if len(sys.argv) == 3:
        out = out[sys.argv[2]]
        print(out.pretty())
    else:
        for fname, func in out.items():
            print('\n====================================')
            print('Function', fname)
            print(func.pretty())
