
import lark

from spatialstencil.syntax import astnodes

class TreeToAST(lark.Transformer):
    # Low-level literal syntax
    digit = lambda self, val: int(val[0])
    digits = lambda self, val: int(val[0])
    hex_digit = lambda self, val: str(val[0])
    hex_digits = lambda self, val: str(val[0])
    letter = lambda self, val: str(val[0])
    letters = lambda self, val: str(val[0])
    underscore = lambda self, val: str(val[0])
    true = lambda self, _: True
    false = lambda self, _: False

    # Literals
    @lark.v_args(inline=True)
    def decimal_literal(self, *digits):
        return int(''.join(str(d) for d in digits))

    @lark.v_args(inline=True)
    def hexadecimal_literal(self, *digits):
        return '0x' + ''.join(digits)

    negated_integer_literal = lambda self, value: -value[0]
    float_literal = lambda self, value: float(value[0])

    @lark.v_args(inline=True)
    def string_literal(self, s):
        return astnodes.StringLiteral(s[1:-1].replace('\\"', '"'))

    @lark.v_args(inline=True)
    def bare_id(self, *elements):
        return ''.join(str(s) for s in elements)

    @lark.v_args(inline=True)
    def suffix_id(self, *suffix):
        return ''.join(str(s) for s in suffix)

    # List types
    extent_tuple = tuple
    extent_tuple_list = list
    dim_list = list
    id_list = list
    type_list = list
    attributes = list
    subscript_slice = list

    def value_expr(self, args, meta=None):
        # Contract/inline value expressions that only contain another value expression
        if len(args) == 1 and isinstance(args[0], lark.Tree) and args[0].data == 'value_expr':
            return args[0]
        return lark.Tree('value_expr', args, meta)

    # Basic types
    identifier = astnodes.Identifier.from_lark
