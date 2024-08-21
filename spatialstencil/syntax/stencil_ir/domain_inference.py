import warnings
from typing import Sequence

import spatialstencil.syntax.stencil_ir.irnodes as sast
import copy

def infer_field_domains(program: sast.Program,
                        domain: sast.Cartesian | None = None,
                        def_use: dict[str, set[sast.FieldType]] | None = None):
    """
    Infers the domain size of a Stencil IR program by traversing it backwards.
    Operates in-place.

    :param program: The Stencil IR program to traverse.
    :param domain: An optional 3-tuple representing domain size (x, y, z). If not given, existing domain size will
                   be used or "?" will remain.
    :param def_use: A dictionary of field names to a set of uses (field type objects).
    """
    if domain is None:  # Nothing to do
        return

    # TODO: Using the same domains overall might be flawed?
    field_domains: dict[str, sast.Cartesian] = {}

    # Start with outputs. Use halo for extents.
    assert isinstance(program.operation_type.destination, list)
    for field, dtype in zip(program.outputs, program.operation_type.destination):
        if dtype.domain.is_unknown():
            dtype.domain = domain
        field_domains[field.name] = dtype.domain

    #for field, dtype in zip(program.inputs, program.operation_type.source):
        #field_domains[field.name] = _infer_domain_from_extents(domain, dtype.extent)
        #print(field.name, field_domains[field.name])

    # Gather failed identifiers for warnings
    potentially_unknown_identifiers: set[str] = set()

    # Traverse the program backwards and accumulate the extents
    for computation in reversed(program.computations):
        # Initialize the output extents to be (0, 0, 0)
        if isinstance(computation, sast.ComputationBlock):

            for node in reversed(list(computation.walk())):
                if isinstance(node, sast.StatementBlock):
                    stmt_domain = None
                    for out, outtype in zip(node.outputs, node.operation_type.destination):
                        if stmt_domain is None:
                            if not outtype.domain.is_unknown():
                                stmt_domain = outtype.domain
                            elif out.name in field_domains:
                                stmt_domain = field_domains[out.name]
                        else:
                            if not outtype.domain.is_unknown() and outtype.domain != stmt_domain:
                                raise ValueError('Ambiguous domains found when processing multiple '
                                                 f'statement outputs: {stmt_domain} != {outtype.domain}')
                            if out.name in field_domains and field_domains[out.name] != stmt_domain:
                                raise ValueError('Ambiguous domains found when processing multiple '
                                                 f'statement outputs: {stmt_domain} != {field_domains[out.name]}')
                    if stmt_domain is None:
                        for out in node.outputs:
                            potentially_unknown_identifiers.add(out.name)
                        continue

                    # Compute input domains based on extents and output
                    for inp, inptype in zip(node.inputs, node.operation_type.source):
                        new_domain = _infer_domain_from_extents(stmt_domain,
                                                                inptype.extent,
                                                                computation.interval)
                        # Take max value from current domain if in dictionary
                        if inp.name in field_domains:
                            dom = field_domains[inp.name]
                            field_domains[inp.name] = dom.union(new_domain)
                        else:
                            # Else, copy
                            field_domains[inp.name] = copy.deepcopy(new_domain)

                elif isinstance(node, sast.MaterializeOp):
                    if node.result.name not in field_domains:
                        warnings.warn(f'Cannot infer domain size from materialization of "%{node.value.name}"')
                        continue
                    field_domains[node.value.name] = field_domains[node.result.name]

    # Warn on still-unknown identifiers
    for identifier in potentially_unknown_identifiers:
        if identifier not in field_domains:
            warnings.warn(f'Could not infer domain size for "%{identifier}"')

    # Assign inferred domain sizes across Stencil IR program
    DomainAssigner(field_domains).visit(program)


def _infer_domain_from_extents(output_domain: sast.Cartesian,
                               extents: sast.Extent,
                               intervals: Sequence[sast.Interval]) -> sast.Cartesian:
    """
    Given the output domain and extents, infers the domain size of the input field.
    Assuming that the output is accessed at offset (0, 0, 0).
    :param output_domain:
    :param extents:
    :return:
    """
    assert len(intervals) == 3
    current_domain = copy.deepcopy(output_domain)

    for e in extents.extents:
        # Convert the extent interval into a cartesian domain representing the output
        extent_domain = output_domain.intersect_with_ranges(intervals)
        # Expand the domain with the extent values
        extent_domain = extent_domain.add(e.values)

        current_domain = current_domain.union(extent_domain)

    return current_domain


class DomainAssigner(sast.NodeTransformer):
    """
    Sets domain based on given dictionary.
    """

    def __init__(self, field_domains: dict[str, sast.Cartesian]):
        super().__init__()
        self.field_domains = field_domains

    def _modify_typeinfo(self, operation_type: sast.OperationType, inputs: list[sast.Identifier],
                         outputs: list[sast.Identifier]):
        """
        Helper function that updates the type information based on inferred types.
        """
        for name, src in zip(inputs, operation_type.source):
            if name.name in self.field_domains and isinstance(src, sast.FieldType) and src.domain.is_unknown():
                src.domain = copy.deepcopy(self.field_domains[name.name])

        if operation_type.destination:
            for name, dst in zip(outputs, operation_type.destination):
                if name.name in self.field_domains and isinstance(dst, sast.FieldType) and dst.domain.is_unknown():
                    dst.domain = copy.deepcopy(self.field_domains[name.name])

    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        self._modify_typeinfo(node.operation_type, [node.value], [node.result])
        return self.generic_visit(node)

    def visit_StatementBlock(self, node: sast.StatementBlock):
        self._modify_typeinfo(node.operation_type, node.inputs, node.outputs)
        return self.generic_visit(node)

    def visit_IfBlock(self, node: sast.IfBlock):
        self._modify_typeinfo(node.operation_type, [node.condition], node.outputs)
        return self.generic_visit(node)

    def visit_ComputationBlock(self, node: sast.ComputationBlock):
        self._modify_typeinfo(node.operation_type, node.inputs, node.outputs)
        return self.generic_visit(node)

    def visit_Program(self, node: sast.Program):
        self._modify_typeinfo(node.operation_type, node.inputs, node.outputs)
        return self.generic_visit(node)
