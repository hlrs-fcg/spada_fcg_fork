import unittest
from pathlib import Path

from spatialstencil.syntax.stencil_ir import type_inference, parser, canonicalization
from spatialstencil.syntax.stencil_ir.irnodes import ScalarType, Program


class TestTypeInference(unittest.TestCase):

    def test_result_type_simple(self):
        # Test unary, binary, and ternary self-comparisons
        for dtype in ScalarType:
            assert type_inference._result_type_of(dtype) == dtype
            assert type_inference._result_type_of(dtype, dtype) == dtype
            assert type_inference._result_type_of(dtype, dtype, dtype) == dtype

    def test_result_type_expansion(self):
        assert type_inference._result_type_of(ScalarType.u16, ScalarType.i32) == ScalarType.i32
        assert type_inference._result_type_of(ScalarType.bool, ScalarType.i16) == ScalarType.i16
        assert type_inference._result_type_of(ScalarType.f64, ScalarType.bool) == ScalarType.f64
        assert type_inference._result_type_of(ScalarType.f64, ScalarType.f32) == ScalarType.f64
        assert type_inference._result_type_of(ScalarType.f16, ScalarType.i32) == ScalarType.f16
        assert type_inference._result_type_of(ScalarType.i16, ScalarType.u16) == ScalarType.u16

    def test_result_type_function(self):
        assert type_inference._result_type_of(ScalarType.bool, optype='sqrt') == ScalarType.f16
        assert type_inference._result_type_of(ScalarType.i32, optype='cbrt') == ScalarType.f32

    def test_infer_expression(self):
        # Parse a simple program
        program = parser.parse_string('''
        %out = spst.program (%inp, %shortinp) {} : 
          field<domain<?, ?, ?>, extent<(?, ?, ?)>, i32>,
          field<domain<?, ?, ?>, extent<(?, ?, ?)>, i16> ->
          field<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> {
            %out = spst.computation(%inp) {
              schedule = PARALLEL,
              interval = [interval<?, ?>, interval<?, ?>, interval<?, ?>]
            } : field<domain<?, ?, ?>, extent<(?, ?, ?)>, i32>,
                field<domain<?, ?, ?>, extent<(?, ?, ?)>, i16> ->
                field<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> {
                    %out = spst.statement (%inp) {} :
                      spst.field<spst.cartesian<?, ?, ?>, spst.extent<(?, ?, ?)>, i32>,
                      spst.field<spst.cartesian<?, ?, ?>, spst.extent<(?, ?, ?)>, i16> ->
                      spst.field<spst.cartesian<?, ?, ?>, spst.extent<(?, ?, ?)>, f32> {
                          %a = -%inp * 0.25
                          %b = %inp[0, 0, 0] + %inp[-1, 0, 0]
                          %c = %shortinp[0, 0, 0] + %shortinp[-1, 0, 0]
                          %d = %inp[0, 0, 0] if %inp else %shortinp[-1, 0, 0]
                          spst.return sqrt(%d) : f32
                    }  
            }
        }
        ''')
        exprs = [ex.value for ex in program.computations[0].body[0].body[:-1]]  # assignments
        exprs += [program.computations[0].body[0].body[-1].values[0]]  # spst.return

        # Missing field
        with self.assertRaises(KeyError):
            type_inference._infer_expression(exprs[0], {'shortinp': ScalarType.i16}, ScalarType.f32, ScalarType.i32)

        fields = {'inp': ScalarType.i32, 'shortinp': ScalarType.i16}

        # Multiplying with a literal
        assert type_inference._infer_expression(exprs[0], fields, ScalarType.f32, ScalarType.i32) == ScalarType.f32
        assert type_inference._infer_expression(exprs[0], fields, ScalarType.f16, ScalarType.i32) == ScalarType.f16

        assert type_inference._infer_expression(exprs[1], fields, ScalarType.f32, ScalarType.i32) == ScalarType.i32
        assert type_inference._infer_expression(exprs[2], fields, ScalarType.f32, ScalarType.i32) == ScalarType.i16
        assert type_inference._infer_expression(exprs[3], fields, ScalarType.f32, ScalarType.i32) == ScalarType.i32

        fields['d'] = ScalarType.i32
        assert type_inference._infer_expression(exprs[4], fields, ScalarType.f32, ScalarType.i32) == ScalarType.f32

    def test_infer_extent_materialize(self):
        file = Path(__file__).parent / Path('../samples/spst/laplacian_mat_sh.spst')
        file2 = Path(__file__).parent / Path('../samples/spst/laplacian_mat_sh_ext.spst')
        program: Program
        program_w_extents: Program
        with open(file, 'r') as f:
            program = parser.parse_file(f)
        with open(file2, 'r') as f:
            program_w_extents = parser.parse_file(f)

        # Infer extents for the program without extents
        type_inference.infer_field_extents(program)

        # Canonicalize program with extents
        program_w_extents = canonicalization.canonicalize(program_w_extents)

        # Check that the extents are the same
        # Note that this also checks the canonicalization of extents
        self.assertEqual(program.as_ir(), program_w_extents.as_ir())

    def test_infer_extent_no_materialize(self):
        file = Path(__file__).parent / Path('../samples/spst/laplacian_no_mat.spst')
        file2 = Path(__file__).parent / Path('../samples/spst/laplacian_no_mat_ext.spst')
        program: Program
        program_w_extents: Program
        with open(file, 'r') as f:
            program = parser.parse_file(f)
        with open(file2, 'r') as f:
            program_w_extents = parser.parse_file(f)

        # Infer extents for the program without extents
        type_inference.infer_field_extents(program)

        # Canonicalize program with extents
        program_w_extents = canonicalization.canonicalize(program_w_extents)

        # Check that the extents are the same
        # Note that this also checks the canonicalization of extents
        self.assertEqual(program.as_ir(), program_w_extents.as_ir())

    def test_canonicalize_extents(self):
        # Parse a simple program
        program = parser.parse_string('''
        %b = spst.program (%a) {} : 
          field<[?, ?, ?], {(0, 0, 0), (0, -1, 0) in [0:None, 0:None, 1:-1], (0, 0, 0)}, f32> ->
          field<[?, ?, ?], {(?, ?, ?)}, f32> {
            %b = spst.computation(%a) {
              schedule = PARALLEL,
              interval = [0:None, 0:None, 0:None]
            } : field<[?, ?, ?], {(0, 0, 0), (0, -1, 0) in [0:None, 0:None, 1:-1], (0, 0, 0)}, f32> ->
                field<[?, ?, ?], {(?, ?, ?)}, f32> {
                    %b = spst.statement (%a) {} :
                      field<[?, ?, ?], {(0, 0, 0), (-1, 0, 0)}, f32> -> field<[?, ?, ?], {(0, 0, 0)}, f32> {
                          spst.return %a : f32
                    }  
            }
        }
        ''')
        non_canonical = '''%b = spst.program(%a) {} : spst.field<spst.cartesian<?, ?, ?>, spst.extent<(0, 0, 0), (0, -1, 0) in [0:None, 0:None, 1:-1], (0, 0, 0)>, f32> -> spst.field<spst.cartesian<?, ?, ?>, spst.extent<(?, ?, ?)>, f32> {
  %b = spst.computation (%a) {
   schedule = PARALLEL,
   interval = [spst.interval<0, None>, spst.interval<0, None>, spst.interval<0, None>]
  } : spst.field<spst.cartesian<?, ?, ?>, spst.extent<(0, 0, 0), (0, -1, 0) in [0:None, 0:None, 1:-1], (0, 0, 0)>, f32> -> spst.field<spst.cartesian<?, ?, ?>, spst.extent<(?, ?, ?)>, f32> {
    %b = spst.statement (%a) {} : spst.field<spst.cartesian<?, ?, ?>, spst.extent<(0, 0, 0), (-1, 0, 0)>, f32> -> spst.field<spst.cartesian<?, ?, ?>, spst.extent<(0, 0, 0)>, f32> {
      spst.return %a : f32
    }
  }
}'''
        canonical = '''%b = spst.program(%a) {} : spst.field<spst.cartesian<?, ?, ?>, spst.extent<(0, 0, 0), (0, -1, 0) in [0:None, 0:None, 1:-1]>, f32> -> spst.field<spst.cartesian<?, ?, ?>, spst.extent<(?, ?, ?)>, f32> {
  %b = spst.computation (%a) {
   schedule = PARALLEL,
   interval = [spst.interval<0, None>, spst.interval<0, None>, spst.interval<0, None>]
  } : spst.field<spst.cartesian<?, ?, ?>, spst.extent<(0, 0, 0), (0, -1, 0) in [0:None, 0:None, 1:-1]>, f32> -> spst.field<spst.cartesian<?, ?, ?>, spst.extent<(?, ?, ?)>, f32> {
    %b = spst.statement (%a) {} : spst.field<spst.cartesian<?, ?, ?>, spst.extent<(-1, 0, 0), (0, 0, 0)>, f32> -> spst.field<spst.cartesian<?, ?, ?>, spst.extent<(0, 0, 0)>, f32> {
      spst.return %a : f32
    }
  }
}'''
        assert program.as_ir() == non_canonical

        cprogram = canonicalization.canonicalize(program)
        assert cprogram.as_ir() == canonical


if __name__ == '__main__':
    unittest.main()
