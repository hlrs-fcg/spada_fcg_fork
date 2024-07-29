import unittest
from spatialstencil.syntax.stencil_ir import parser, astnodes as stast
import os


class TestStencilIRParser(unittest.TestCase):

    def test_if_block(self):
        """
        Tests parsing an if block
        """
        src = '''
        %out = spst.program (%inp) {} : 
          field<domain<?, ?, ?>, extent<(?, ?, ?)>, bool> ->
          field<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> {
            %out = spst.computation(%inp) {
              schedule = PARALLEL,
              interval = [interval<?, ?>, interval<?, ?>, interval<?, ?>]
            } : field<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> ->
                field<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> {                    
                    %b = spst.if (%inp) : field<domain<?, ?, ?>, extent<(?, ?, ?)>, bool> -> field<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> {
                        spst.return %inp
                    } elif (%arg1) {
                        spst.return %inp + 1 : field<domain<?, ?, ?>, extent<(?, ?, ?)>, f32>
                    } else {
                        spst.return 0;
                    }
                    %out = spst.if (%b) : field<domain<?, ?, ?>, extent<(?, ?, ?)>, bool> -> field<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> {
                        spst.return %inp
                    } else {
                        spst.return 0;
                    }
            }
        }
        '''
        program = parser.parse_string(src)
        comp = program.computations[0]

        # Branch 1
        assert isinstance(comp.body[0], stast.IfBlock)
        assert comp.body[0].result.as_ir() == '%b'
        assert comp.body[0].condition.as_ir() == '%inp'
        assert len(comp.body[0].else_ifs) == 1
        assert comp.body[0].orelse is not None

        # Branch 2
        assert isinstance(comp.body[1], stast.IfBlock)
        assert comp.body[1].result.as_ir() == '%out'
        assert comp.body[1].condition.as_ir() == '%b'
        assert len(comp.body[1].else_ifs) == 0
        assert comp.body[1].orelse is not None

    def test_roundtrip_hdiff(self):
        """
        Tests a roundtrip IR->parse->IR->parse->IR for differences.
        """
        file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'spst', 'hdiff.spst')
        program = parser.parse_file(file)
        ir_1 = program.as_ir()
        program2 = parser.parse_string(ir_1)
        ir_2 = program2.as_ir()
        assert ir_1 == ir_2

    def test_roundtrip_vadv(self):
        """
        Tests a roundtrip IR->parse->IR->parse->IR for differences.
        """
        file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'spst', 'vadv.spst')
        program = parser.parse_file(file)
        ir_1 = program.as_ir()
        program2 = parser.parse_string(ir_1)
        ir_2 = program2.as_ir()
        assert ir_1 == ir_2


if __name__ == '__main__':
    unittest.main()
