import unittest
from spatialstencil.syntax.gt4py import parser
from spatialstencil.lowering import gt4py_to_stencil_ir
import numpy as np

Field3D = np.ndarray


class TestStencilIRParser(unittest.TestCase):

    def test_lower_gt4py_if(self):
        gtfuncs = parser.parse_file(__file__)  # Parse this file
        program = gtfuncs['satadjust_specific_humidity']
        self.skipTest('Predication pass not yet implemented')
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)

    def test_lower_gt4py_nested_if(self):
        gtfuncs = parser.parse_file(__file__)  # Parse this file
        program = gtfuncs['satadjust_specific_humidity_nestedif']
        self.skipTest('Predication pass not yet implemented')
        with self.assertRaises(SyntaxError):
            gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)


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
