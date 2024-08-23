import unittest
from pathlib import Path

from spatialstencil.syntax.stencil_ir import type_inference, parser, canonicalization, extent_inference, \
    domain_inference
from spatialstencil.syntax.stencil_ir.irnodes import ScalarType, Program, Cartesian, Interval, Offset, Extent, \
    StatementBlock, MaterializeOp, ComputationBlock, ReturnOp


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


    def test_infer_domain_from_extents(self):

        output_domain = Cartesian.from_sequence((0, 128, 0, 128, 0, 80))

        extents = [
            Offset((0, 1, 0)),
            Offset((-1, 0, 0)),
            Offset((0, 0, 2))
        ]
        extent = Extent(extents)

        golden_result = Cartesian.from_sequence((-1, 128, 0, 129, 0, 80))

        intervals = [Interval(0, None), Interval(0, None), Interval(0, -2)]

        result = domain_inference._infer_domain_from_extents(output_domain,
                                                             extent,
                                                             intervals)

        assert result == golden_result


    def test_infer_domain_from_extents_2(self):

        output_domain = Cartesian.from_sequence((0, 128, 0, 128, 0, 80))

        extents = [
            Offset((0, 0, -5)),
            Offset((0, 0, 2))
        ]
        extent = Extent(extents)

        # Test 1
        golden_result = Cartesian.from_sequence((0, 128, 0, 128, 0, 80))

        intervals = [Interval(0, None), Interval(0, None), Interval(-5, -2)]


        result = domain_inference._infer_domain_from_extents(output_domain,
                                                             extent,
                                                             intervals)

        assert result == golden_result

        # Test 2
        golden_result = Cartesian.from_sequence((0, 128, 0, 128, 0, 80))

        intervals = [Interval(0, None), Interval(0, None), Interval(5, 78)]

        result = domain_inference._infer_domain_from_extents(output_domain,
                                                             extent,
                                                             intervals)

        assert result == golden_result


    def assert_infer_domains(self, program: Program):
        """
        Asserts that all domains have been inferred
        :param program:
        :return:
        """
        domain_inference.infer_field_domains(program, Cartesian.from_sequence((0, 128, 0, 128, 0, 80)))

        print(program.as_ir())

        # Check that all domains have been inferred
        for computation in program.computations:
            if isinstance(computation, ComputationBlock):
                for statement in computation.body:
                    for input in statement.operation_type.source:
                        if isinstance(input, ScalarType):
                            continue
                        assert not input.domain.is_unknown()
                    if isinstance(statement, StatementBlock) or isinstance(statement, MaterializeOp):
                        for output in statement.operation_type.destination:
                            assert not output.domain.is_unknown()
                for output in computation.operation_type.source:
                    if isinstance(output, ScalarType):
                        continue
                    assert not output.domain.is_unknown()
                for input in computation.operation_type.destination:
                    if isinstance(input, ScalarType):
                        continue
                    assert not input.domain.is_unknown()
            elif isinstance(computation, ReturnOp):
                for output in computation.operation_type.source:
                    assert not output.domain.is_unknown()


    def test_infer_domains_is_complete(self):
        # For every file, run the parser, infer_domains
        # Check that all domains have been inferred (i.e. no unknown domains)

        files = [
            Path(__file__).parent / Path('../samples/spst/laplacian_mat_ext.spst'),
            Path(__file__).parent / Path('../samples/spst/laplacian_no_mat_ext.spst'),
            Path(__file__).parent / Path('../samples/spst/if_else_ext.spst'),
            Path(__file__).parent / Path('../samples/spst/multiple_returns_ext.spst'),
            Path(__file__).parent / Path('../samples/spst/laplacian_mat_sh_ext.spst')
        ]

        for file in files:
            with open(file, 'r') as f:
                program = parser.parse_file(f)
            print(program.as_ir())

            self.assert_infer_domains(program)


    def test_infer_extents_and_domains(self):

        files = [
            Path(__file__).parent / Path('../samples/spst/hdiff.spst'),
            Path(__file__).parent / Path('../samples/spst/vadv.spst')
        ]

        for file in files:
            with open(file, 'r') as f:
                program = parser.parse_file(f)
            print(program.as_ir())

            # extent inference
            extent_inference.infer_field_extents(program)
            print(program.as_ir())

            # Assert all statements have extents
            for computation in program.computations:
                if isinstance(computation, ComputationBlock):
                    for statement in computation.body:
                        if isinstance(statement, StatementBlock) or isinstance(statement, MaterializeOp):
                            for input in statement.operation_type.source:
                                if isinstance(input, ScalarType):
                                    continue
                                assert not input.extent.is_unknown()
                            for output in statement.operation_type.destination:
                                assert not output.extent.is_unknown()
                elif isinstance(computation, ReturnOp):
                    for output in computation.operation_type.source:
                        assert not output.extent.is_unknown()

            # domain inference
            self.assert_infer_domains(program)


    def test_infer_expression(self):
        # Parse a simple program
        program = parser.parse_string('''
        %out = spst.program (%inp, %shortinp) {} : 
          field<domain<?, ?, ?>, extent<(?, ?, ?)>, i32>,
          field<domain<?, ?, ?>, extent<(?, ?, ?)>, i16> ->
          field<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> {
            %out = spst.computation(%inp, %shortinp) {
              schedule = PARALLEL,
              interval = [interval<?, ?>, interval<?, ?>, interval<?, ?>]
            } : field<domain<?, ?, ?>, extent<(?, ?, ?)>, i32>,
                field<domain<?, ?, ?>, extent<(?, ?, ?)>, i16> ->
                field<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> {
                    %out = spst.statement (%inp, %shortinp) {} :
                      spst.field<spst.cartesian<?, ?, ?>, spst.extent<(?, ?, ?)>, i32>,
                      spst.field<spst.cartesian<?, ?, ?>, spst.extent<(?, ?, ?)>, i16> ->
                      spst.field<spst.cartesian<?, ?, ?>, spst.extent<(?, ?, ?)>, f32> {
                          %a = -%inp * 0.25
                          %b = %inp[0, 0, 0] + %inp[-1, 0, 0]
                          %c = %shortinp[0, 0, 0] + %shortinp[-1, 0, 0]
                          %d = %inp[0, 0, 0] if %inp else %shortinp[-1, 0, 0]
                          spst.return sqrt(%d) : f32
                    }
                    spst.return %out: field<[?,?,?], {(?, ?, ?)}, f32>
            }
            spst.return %out: field<[?,?,?], {(?, ?, ?)}, f32>
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

    def _test_programs_equal(self, file1, file2):
        test_program: Program
        golden_program: Program
        with open(file1, 'r') as f:
            test_program = parser.parse_file(f)
        with open(file2, 'r') as f:
            golden_program = parser.parse_file(f)

        print(test_program.as_ir())
        # Infer extents for the program without extents
        extent_inference.infer_field_extents(test_program)

        # Canonicalize program with extents
        golden_program = canonicalization.canonicalize(golden_program)

        print(test_program.as_ir())

        # Check the overall program
        self.assertEqual(golden_program.as_ir(), test_program.as_ir())

    def test_infer_extent_ifelse(self):
        file = Path(__file__).parent / Path('../samples/spst/if_else.spst')
        file2 = Path(__file__).parent / Path('../samples/spst/if_else_ext.spst')

        self._test_programs_equal(file, file2)


    def test_multiple_returns(self):
        file = Path(__file__).parent / Path('../samples/spst/multiple_returns.spst')
        file2 = Path(__file__).parent / Path('../samples/spst/multiple_returns_ext.spst')

        self._test_programs_equal(file, file2)

    def test_infer_extent_materialize(self):
        file = Path(__file__).parent / Path('../samples/spst/laplacian_mat_sh.spst')
        file2 = Path(__file__).parent / Path('../samples/spst/laplacian_mat_sh_ext.spst')

        self._test_programs_equal(file, file2)

    def test_infer_extent_no_materialize(self):
        file = Path(__file__).parent / Path('../samples/spst/laplacian_no_mat.spst')
        file2 = Path(__file__).parent / Path('../samples/spst/laplacian_no_mat_ext.spst')

        self._test_programs_equal(file, file2)


    def test_infer_domain_from_extents(self):

        file = Path(__file__).parent / Path('../samples/spst/laplacian_mat_sh.spst')
        file2 = Path(__file__).parent / Path('../samples/spst/laplacian_mat_ext_dom.spst')

        with open(file, 'r') as f:
            test_program = parser.parse_file(f)

        extent_inference.infer_field_extents(test_program)
        domain_inference.infer_field_domains(test_program, Cartesian.from_sequence((0, 128, 0, 128, 0, 80)))
        test_program = canonicalization.canonicalize(test_program)

        with open(file2, 'r') as f:
            golden_program = parser.parse_file(f)

        golden_program = canonicalization.canonicalize(golden_program)

        self.assertEqual(golden_program.as_ir(), test_program.as_ir())

    def test_canonicalize_extents(self):
        # Parse a simple program
        program = parser.parse_string('''
        %b = spst.program (%a) {} : 
          field<[?, ?, ?], {(0, 0, 0), (0, -1, 0), (0, 0, 0)}, f32> ->
          field<[?, ?, ?], {(?, ?, ?)}, f32> {
            %b = spst.computation(%a) {
              schedule = PARALLEL,
              interval = [0:None, 0:None, 0:None]
            } : field<[?, ?, ?], {(0, 0, 0), (0, -1, 0), (0, 0, 0)}, f32> ->
                field<[?, ?, ?], {(?, ?, ?)}, f32> {
                    %b = spst.statement (%a) {} :
                      field<[?, ?, ?], {(0, 2, 0), (1, 0, 0), (?, ?, 2), (-1, 0, 1)}, f32> -> field<[?, ?, ?], {(0, 0, 0)}, f32> {
                          spst.return %a : f32
                    }
                spst.return %b : field<[?:?, ?:?, ?:?], {(?, ?, ?)}, f32>
            }
            spst.return %b : field<[?:?, ?:?, ?:?], {(?, ?, ?)}, f32>
        }
        ''')

        canonical = '''%b = spst.program(%a) {} : spst.field<[?:?, ?:?, ?:?], {(0, -1, 0), (0, 0, 0)}, f32> -> spst.field<[?:?, ?:?, ?:?], {(?, ?, ?)}, f32> {
  %b = spst.computation (%a) {
   schedule = PARALLEL,
   interval = [0:None, 0:None, 0:None]
  } : spst.field<[?:?, ?:?, ?:?], {(0, -1, 0), (0, 0, 0)}, f32> -> spst.field<[?:?, ?:?, ?:?], {(?, ?, ?)}, f32> {
    %b = spst.statement (%a) {} : spst.field<[?:?, ?:?, ?:?], {(?, ?, 2), (-1, 0, 1), (0, 2, 0), (1, 0, 0)}, f32> -> spst.field<[?:?, ?:?, ?:?], {(0, 0, 0)}, f32> {
      spst.return %a : f32
    }
    spst.return %b : spst.field<[?:?, ?:?, ?:?], {(?, ?, ?)}, f32>
  }
  spst.return %b : spst.field<[?:?, ?:?, ?:?], {(?, ?, ?)}, f32>
}'''

        cprogram = canonicalization.canonicalize(program)
        print(cprogram.as_ir())
        assert cprogram.as_ir() == canonical


if __name__ == '__main__':
    unittest.main()
