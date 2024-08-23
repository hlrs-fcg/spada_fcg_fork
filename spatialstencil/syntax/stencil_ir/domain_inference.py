import warnings
from typing import Sequence, Collection

import spatialstencil.syntax.stencil_ir.irnodes as sast
import copy

from spatialstencil.syntax.stencil_ir import def_use_analysis
from spatialstencil.syntax.stencil_ir.def_use_analysis import ScopedUse


def infer_field_domains(program: sast.Program,
                        domain: sast.Cartesian | None = None,
                        def_use: dict[sast.Identifier, list[ScopedUse]] | None = None):
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

    if def_use is None:
        def_use = dict()
        def_use_analysis.DefUseAnalysis(def_use).visit(program)

    dom_inference = DomainInference(def_use, domain)
    dom_inference.visit(program)

    # Warn on still-unknown identifiers
    #for identifier in potentially_unknown_identifiers:
    #    if identifier not in field_domains:
    #        warnings.warn(f'Could not infer domain size for "%{identifier}"')

    # Assign inferred domain sizes across Stencil IR program
    #print(field_domains)
    #DomainAssigner(field_domains).visit(program)

def _union_of_domains_of_uses_in_scope(uses: dict[sast.Identifier, Collection[def_use_analysis.ScopedUse]],
                                       computation: sast.ComputationBlock,
                                       identifier: sast.Identifier) -> sast.Cartesian:
    return _union_domains(_domains_of_uses_in_scope(uses, computation, identifier))


def _union_domains(domains: list[sast.Domain]) -> sast.Cartesian:
    """
    Given a list of domains, returns the union of all domains.
    """
    assert len(domains) > 0
    dom = domains[0]
    for d in domains[1:]:
        dom = dom.union(d)
    return dom


def _domains_of_uses_in_scope(uses: dict[sast.Identifier, Collection[def_use_analysis.ScopedUse]],
                              computation: sast.ComputationBlock | sast.Program,
                              identifier: sast.Identifier) -> list[sast.Cartesian]:
    """
    Get the offsets of the uses of a field in the current scope.
    :param uses: The uses dictionary
    :param computation: The current computation block
    :param identifier: The identifier
    :return: A list of offsets
    """
    assert isinstance(identifier, sast.Identifier)

    uses_of_result = uses.get(identifier) or []
    uses_of_result = [u for u in uses_of_result if u.definition_scope == computation]
    # Concatenate all the extents from uses_of_result
    use_domains = []
    for use in uses_of_result:
        use_domains.append(use.field_type.domain)
    if len(use_domains) == 0:
        print(f"WARNING: No uses of {identifier.name} found within current scope.")

    assert all(not o.is_unknown() for o in use_domains), f"Domain of uses of {identifier.name} must be known before inference"

    return use_domains


def _infer_domain_from_extents(output_domain: sast.Cartesian,
                               extents: sast.Extent,
                               intervals: Sequence[sast.Interval]) -> sast.Cartesian:
    """
    Given the output domain and extents, infers the domain size of the input field.
    Assuming that the output is accessed at offset (0, 0, 0).
    # TODO Adapt for non-zero offsets
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


class DomainInference(sast.ScopedNodeVisitor):

    def __init__(self, def_use: dict[sast.Identifier, list[ScopedUse]], result_domain: sast.Cartesian):
        super().__init__(reverse=True)
        self.domain = result_domain
        self.def_use = def_use

    def __post_init__(self):
        assert self.reverse

    def visit_Program(self, program: sast.Program):
        self.push_scope(program)
        # Start with outputs. Use halo for extents.
        assert isinstance(program.operation_type.destination, list)
        for field, dtype in zip(program.outputs, program.operation_type.destination):
            if dtype.domain.is_unknown():
                dtype.domain = self.domain
        self.generic_visit(program)
        # Set the input domains to the union of the domains of their uses
        for inp, inptype in zip(program.inputs, program.operation_type.source):
            in_domains = _domains_of_uses_in_scope(self.def_use, program, inp)
            in_domain = _union_domains(in_domains)
            if inptype.domain.is_unknown():
                inptype.domain = in_domain
        self.pop_scope()

    def visit_MaterializeOp(self, node: sast.MaterializeOp):
        computation = self.get_scope_with_type(sast.ComputationBlock)
        assert computation
        assert isinstance(computation, sast.ComputationBlock)
        # Initialize materialize op types with the domain of the result
        domain = _union_of_domains_of_uses_in_scope(self.def_use, computation, node.result)
        node.operation_type.destination[0].domain = copy.deepcopy(domain)
        node.operation_type.source[0].domain = copy.deepcopy(domain)

    def visit_ReturnOp(self, node: sast.ReturnOp):
        scope = self.get_scope()
        if isinstance(scope, sast.ComputationBlock):
            # Initialize return op types with the domain of the computation block's result
            for i in range(len(node.values)):
                node.operation_type.source[i].domain = copy.deepcopy(scope.operation_type.destination[i].domain)
        elif isinstance(scope, sast.Program):
            for i in range(len(node.values)):
                node.operation_type.source[i].domain = copy.deepcopy(self.domain)

    def visit_StatementBlock(self, node: sast.StatementBlock):
        computation = self.get_scope_with_type(sast.ComputationBlock)
        assert computation
        assert isinstance(computation, sast.ComputationBlock)
        # The output domains are given by the union of the domains of their uses
        out_domains = []
        for out, outtype in zip(node.outputs, node.operation_type.destination):
            out_domains.extend(_domains_of_uses_in_scope(self.def_use, computation, out))
        out_domain = _union_domains(out_domains)
        # Initialize output domains with the domain of the result
        for outtype in node.operation_type.destination:
            outtype.domain = copy.deepcopy(out_domain)

        # Compute input domains based on extents and output
        for inp, inptype in zip(node.inputs, node.operation_type.source):
            new_domain = _infer_domain_from_extents(out_domain,
                                                    inptype.extent,
                                                    computation.interval)
            # Take max value from current domain if in dictionary
            inptype.domain = inptype.domain.union(new_domain)

    def visit_ComputationBlock(self, node: sast.ComputationBlock):
        # Initialize output domains with the domain of the result
        for out, outtype in zip(node.outputs, node.operation_type.destination):
            if outtype.domain.is_unknown():
                outtype.domain = copy.deepcopy(self.domain)

        self.push_scope(node)
        for child in reversed(node.body):
            self.visit(child)
        # Initialize input domains as the union of the domains of their uses
        for inp, inptype in zip(node.inputs, node.operation_type.source):
            in_domains = _domains_of_uses_in_scope(self.def_use, node, inp)
            in_domain = _union_domains(in_domains)
            if inptype.domain.is_unknown():
                inptype.domain = copy.deepcopy(in_domain)
        self.pop_scope()

    def visit_IfBlock(self, node: sast.IfBlock):
        # We need to infer the domain of the outputs based on the uses of the outputs
        # We will compute the output for the union of the uses.
        computation = self.get_scope_with_type(sast.ComputationBlock)
        assert computation
        assert isinstance(computation, sast.ComputationBlock)

        domains = []
        for out, outtype in zip(node.outputs, node.operation_type.destination):
            out_domain = _domains_of_uses_in_scope(self.def_use, computation, out)
            domains.extend(out_domain)

        out_domain = _union_domains(domains)
        for outtype in node.operation_type.destination:
            outtype.domain = copy.deepcopy(out_domain)

        # Visit the condition, body, and else_ifs
        self.generic_visit(node)

        # The input domain must match the output domain
        # as it serves as a mask for which output to use
        conditions = [node.condition]
        for elifblock in node.else_ifs:
            conditions.append(elifblock.condition)

        for inp, inptype in zip(conditions, node.operation_type.source):
            inptype.domain = out_domain

