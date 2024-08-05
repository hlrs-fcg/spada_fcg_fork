import unittest
from spatialstencil.syntax.gt4py import parser
from spatialstencil.syntax.stencil_ir import astnodes as sast
from spatialstencil.lowering import gt4py_to_stencil_ir
import numpy as np

Field3D = np.ndarray


class TestStencilIRParser(unittest.TestCase):

    def setUp(self):
        self.gtfuncs = parser.parse_file(__file__)  # Parse this file

    def test_lower_gt4py_intermediates(self):
        program = self.gtfuncs['simple']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)
        assert irprogram.name == 'simple'
        assert len(irprogram.inputs) == 1
        assert len(irprogram.outputs) == 1
        assert len(irprogram.computations) == 1

        # Test computation
        comp = irprogram.computations[0]
        assert comp.schedule == sast.ComputationType.PARALLEL
        assert comp.interval == (sast.Interval(0, None), sast.Interval(0, None), sast.Interval(0, None))

    def test_lower_gt4py_if(self):
        program = self.gtfuncs['satadjust_specific_humidity']
        self.skipTest('Predication pass not yet implemented')
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)

    def test_lower_gt4py_nested_if(self):
        program = self.gtfuncs['satadjust_specific_humidity_nestedif']
        self.skipTest('Predication pass not yet implemented')
        with self.assertRaises(SyntaxError):
            gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)


def simple(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        tmp = inp + 1
        out = tmp + 1


# GT4Py stencils for the test
# See https://github.com/NOAA-GFDL/PyFV3/blob/fcaf36e6b3101ec655d4c728d573267c52d42132/pyFV3/stencils/saturation_adjustment.py#L883


# Simplified version without nested conditionals
def satadjust_specific_humidity(tin: Field3D, iqs1: Field3D, wqs1: Field3D, q_cond: Field3D, q_sol: Field3D,
                                qpz: Field3D, rh: Field3D):
    with computation(PARALLEL), interval(...):
        if tin < 233.16:
            qstar = iqs1
        elif tin >= 273.16:
            qstar = wqs1
        else:
            rqi = q_sol / q_cond
            qstar = rqi * iqs1 + (1.0 - rqi) * wqs1

        rh = qpz / qstar


def satadjust_specific_humidity_nestedif(tin: Field3D, iqs1: Field3D, wqs1: Field3D, q_cond: Field3D, q_sol: Field3D,
                                         qpz: Field3D, rh: Field3D):
    with computation(PARALLEL), interval(...):
        if tin < 233.16:  # Homogeneous freezing temperature
            # ice phase
            qstar = iqs1
        elif tin >= 273.16:  # Freezing temperature
            qstar = wqs1
        else:
            if q_cond > 1e-6:
                rqi = q_sol / q_cond
            else:
                rqi = (273.16 - tin) / -40.0
            qstar = rqi * iqs1 + (1.0 - rqi) * wqs1

        rh = qpz / qstar


if __name__ == '__main__':
    unittest.main()
