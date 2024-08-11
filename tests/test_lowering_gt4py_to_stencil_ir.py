import unittest
from spatialstencil.syntax.gt4py import parser
from spatialstencil.syntax.stencil_ir import irnodes as sast, analysis
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
        assert [inp.name for inp in irprogram.inputs] == ['inp']
        assert [out.name for out in irprogram.outputs] == ['out']
        assert len(irprogram.computations) == 1

        # Test computation
        comp = irprogram.computations[0]
        assert comp.schedule == sast.ComputationType.PARALLEL
        assert comp.interval == (sast.Interval(0, None), sast.Interval(0, None), sast.Interval(0, None))

        # Test input/output collection
        assert [inp.name for inp in comp.inputs] == ['inp']
        assert [out.name for out in comp.outputs] == ['out']
        extents = analysis.collect_extents(comp)
        assert extents['inp'] == {(0, 0, 0)}
        assert extents['out'] == {(0, 0, 0)}

    def test_lower_gt4py_intermediates_2(self):
        program = self.gtfuncs['unused']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)
        assert irprogram.name == 'unused'
        assert [inp.name for inp in irprogram.inputs] == ['inp']
        assert [out.name for out in irprogram.outputs] == ['out']
        assert len(irprogram.computations) == 2

        # Test computation
        comps = irprogram.computations
        assert comps[0].schedule == sast.ComputationType.PARALLEL
        assert comps[0].interval == (sast.Interval(0, None), sast.Interval(0, None), sast.Interval(0, None))

        # Test input/output collection
        assert [inp.name for inp in comps[0].inputs] == ['inp']
        assert [out.name for out in comps[0].outputs] == ['used']
        assert [inp.name for inp in comps[1].inputs] == ['used']
        assert [out.name for out in comps[1].outputs] == ['out']

    def test_lower_gt4py_intermediates_3(self):
        program = self.gtfuncs['intermediates_versioning']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)
        assert irprogram.name == 'intermediates_versioning'
        assert [inp.name for inp in irprogram.inputs] == ['inp']
        assert [out.name for out in irprogram.outputs] == ['out']
        assert len(irprogram.computations) == 2

        # Test computation
        comps = irprogram.computations
        assert comps[0].schedule == sast.ComputationType.PARALLEL
        assert comps[0].interval == (sast.Interval(0, None), sast.Interval(0, None), sast.Interval(0, None))

        # Test input/output collection
        assert [inp.name for inp in comps[0].inputs] == ['inp']
        assert [out.name for out in comps[0].outputs] == ['interim']
        assert [inp.name for inp in comps[1].inputs] == ['interim']
        assert [out.name for out in comps[1].outputs] == ['out']
        extents = analysis.collect_extents(comps[0])
        assert extents['inp'] == {(0, -1, 0)}
        assert extents['tmp'] == {(1, 0, 0), (0, 0, 0)}

    def test_lower_gt4py_intermediates_4(self):
        program = self.gtfuncs['intermediates_versioning_2']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)
        assert irprogram.name == 'intermediates_versioning_2'
        assert [inp.name for inp in irprogram.inputs] == ['inp']
        assert [out.name for out in irprogram.outputs] == ['out']
        assert len(irprogram.computations) == 2

        # Test computation
        comps = irprogram.computations
        assert comps[0].schedule == sast.ComputationType.PARALLEL
        assert comps[0].interval == (sast.Interval(0, None), sast.Interval(0, None), sast.Interval(0, None))

        # Test input/output collection
        assert [inp.name for inp in comps[0].inputs] == ['inp']
        assert [out.name for out in comps[0].outputs] == ['interim', 'tmp']
        assert [inp.name for inp in comps[1].inputs] == ['interim', 'tmp']
        assert [out.name for out in comps[1].outputs] == ['out']
        extents = analysis.collect_extents(comps[0])
        assert extents['inp'] == {(0, -1, 0)}
        assert extents['tmp'] == {(1, 0, 0), (0, 0, 0)}

    def test_lower_gt4py_intermediates_5(self):
        program = self.gtfuncs['intermediates_versioning_3']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)
        assert irprogram.name == 'intermediates_versioning_3'
        assert [inp.name for inp in irprogram.inputs] == ['inp']
        assert [out.name for out in irprogram.outputs] == ['out']
        assert len(irprogram.computations) == 2

        # Test computation
        comps = irprogram.computations
        assert comps[0].schedule == sast.ComputationType.PARALLEL
        assert comps[0].interval == (sast.Interval(0, None), sast.Interval(0, None), sast.Interval(0, None))

        # Test input/output collection
        assert comps[0].inputs == [sast.Identifier('inp')]
        assert comps[0].outputs == [sast.Identifier('tmp', version=1)]
        assert comps[1].inputs == [sast.Identifier('tmp', version=1)]
        assert comps[1].outputs == [sast.Identifier('out', version=0)]

    def test_lower_mathcall(self):
        program = self.gtfuncs['simple_mathcall']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)
        comp = irprogram.computations[0]
        assert [inp.name for inp in comp.inputs] == ['inp']
        assert [out.name for out in comp.outputs] == ['out']

    def test_output_overwrite(self):
        program = self.gtfuncs['output_overwrite']
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)
        comp = irprogram.computations[0]
        assert comp.inputs == [sast.Identifier('inp'), sast.Identifier('out')]
        assert comp.outputs == [sast.Identifier('out', version=1)]

    def test_lower_gt4py_if(self):
        program = self.gtfuncs['satadjust_specific_humidity']
        self.skipTest('Predication pass not yet implemented')
        irprogram = gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)

    def test_lower_gt4py_nested_if(self):
        program = self.gtfuncs['satadjust_specific_humidity_nestedif']
        self.skipTest('Predication pass not yet implemented')
        with self.assertRaises(SyntaxError):
            gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(program)


# GT4Py stencils for the test
def simple(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        tmp = inp + 1
        out = tmp + 1


def unused(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        used = inp + 1
        unused = inp + 2

    with computation(PARALLEL), interval(...):
        out = used + 1


def intermediates_versioning(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        tmp = inp[0, -1, 0] + 1
        interim = tmp[1, 0, 0] + 1

    with computation(PARALLEL), interval(...):
        tmp = interim + 1
        out = tmp + 1


def intermediates_versioning_2(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        tmp = inp[0, -1, 0] + 1
        interim = tmp[1, 0, 0] + 1

    with computation(PARALLEL), interval(...):
        tmp = interim + tmp
        out = tmp + 1


def intermediates_versioning_3(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        tmp = inp + 1
        tmp = tmp + 1

    with computation(PARALLEL), interval(...):
        tmp = tmp + 1
        out = tmp + 1


def simple_mathcall(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        out = sqrt(inp[1, 0, 0] + 1)


def output_overwrite(inp: Field3D, out: Field3D):
    with computation(PARALLEL), interval(...):
        tmp = inp + out
        out = tmp + 1


# Adapted saturation adjustment subset code from PyFV3, see:
# https://github.com/NOAA-GFDL/PyFV3/blob/fcaf36e6b3101ec655d4c728d573267c52d42132/pyFV3/stencils/saturation_adjustment.py#L883


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
