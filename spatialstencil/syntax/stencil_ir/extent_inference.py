from collections import defaultdict
from typing import Sequence, Collection

import spatialstencil.syntax.stencil_ir.irnodes as sast
import spatialstencil.syntax.stencil_ir.def_use_analysis as def_use_analysis


def infer_field_extents(program: sast.Program):
    """
    Uses def-use analysis to infer the extents of a Stencil IR program by traversing it twice.

    First, we identify for each field where it is used and note that type object.
    This is done using a visitor that creates a dictionary of field names to a set of uses (field type objects).
    The field type objects collected are modified in place!

    Then, we compute the extents for each field based on the uses,
    by traversing the program backwards and accumulating the extents.
    Specifically, the extent of an input to a computation or program is the union of all its uses.
    The extent of a statement argument is the minkowski sum of its 'local' extent and the extents of its uses.

    The 'local' extents are computed using an ExtentCollector.

    :param program:
    :return:
    """

    # Create the def-use analysis object
    uses = dict()
    def_use = def_use_analysis.DefUseAnalysis(uses)
    def_use.visit(program)

    def init_outputs(dtypes: Sequence[sast.FieldType]):
        """
        Initialize the extents of the outputs to be (0, 0, 0)
        :param dtypes:
        :return:
        """
        for dtype in dtypes:
            if dtype.extent.is_unknown():
                dtype.extent.extents = [sast.Offset((0, 0, 0))]

    # Start with outputs. Extents always start at (0, 0, 0)
    init_outputs(program.operation_type.destination)

    # Traverse the program backwards and accumulate the extents
    for computation in reversed(program.computations):
        # Initialize the output extents to be (0, 0, 0)
        if isinstance(computation, sast.ComputationBlock):
            init_outputs(computation.operation_type.destination)

            for node in reversed(list(computation.walk())):
                if isinstance(node, sast.StatementBlock):
                    _walk_StatementBlock(uses, computation, node)
                elif isinstance(node, sast.IfBlock):
                    _walk_IfBlock(uses, computation, node)
                elif isinstance(node, sast.ReturnOp):
                    _walk_ReturnOp(uses, computation, node)
                elif isinstance(node, sast.MaterializeOp):
                    _walk_MaterializeOp(uses, computation, node)


class LocalExtentCollector(sast.NodeVisitor):

    """
    A node visitor that collects all input and output extents from field accesses in the visited blocks/statements.
    """

    # The extents are stored in a dictionary with the field name as key and a set of extents as value.
    extents: dict[str, set[tuple[int]]]

    def __init__(self):
        super().__init__()
        self.extents: dict[str, set[sast.Offset]] = defaultdict(set)

    def visit_StatementBlock(self, node: sast.StatementBlock):
        # Visit only the body (arguments do not count as accesses)
        for b in node.body:
            self.visit(b)
    def visit_Identifier(self, node: sast.Identifier):
        # If a bare identifier (i.e., no subscript) is used, the extent (0, 0, 0) should be added
        self.extents[node.name].add(sast.Offset((0, 0, 0)))

    def visit_Subscript(self, node: sast.Subscript):
        # If a subscript is found, add its subscript to the extents.
        # Make sure not to recursively visit into the subscript to avoid adding (0, 0, 0)
        self.extents[node.value.name].add(sast.Offset((node.subscript[0], node.subscript[1], node.subscript[2])))


def _walk_MaterializeOp(uses: dict[sast.Identifier, list[def_use_analysis.ScopedUse]],
                        computation: sast.ComputationBlock,
                        node: sast.MaterializeOp) -> None:
    # Input is 0 0 0
    node.operation_type.source[0].extent.extents[:] = [sast.Offset((0, 0, 0))]
    # Output is given by the union of the uses.

    node.operation_type.destination[0].extent.extents[:] = _offsets_of_uses_in_scope(uses,
                                                                                     computation,
                                                                                     node.result)
    node.operation_type.destination[0].extent.sort_extents()


def _walk_StatementBlock(uses: dict[sast.Identifier, list[def_use_analysis.ScopedUse]],
                         computation: sast.ComputationBlock,
                         node: sast.StatementBlock) -> None:
    # Get all the uses of the result within the current scope

    # We assume that all outputs are uniform, meaning that the uses are given by the union
    # of the uses of the outputs.
    use_offsets = []
    for output in node.outputs:
        offsets = _offsets_of_uses_in_scope(uses, computation, output)
        use_offsets.extend(offsets)
        #print(f"Use offsets: {offsets} for {output.name}")

    use_offsets = list(set(use_offsets))
    #print(f"Use offsets: {use_offsets}")

    # Compute local extents
    local_extent_collector = LocalExtentCollector()
    local_extent_collector.visit(node)

    # For each input, compute the minkowski sum of the local extent and the uses
    for arg, arg_t in zip(node.inputs, node.operation_type.source):
        # Compute the minkowski sum of the local extent and the uses
        local_extent = local_extent_collector.extents[arg.name]

        #print(f"Local extent for {arg.name}: {local_extent}")

        #print(f"Use offsets for {arg.name}: {use_offsets}")
        if len(use_offsets) > 0:
            arg_t.extent.extents = [*_minkowski_sum(local_extent, use_offsets)]
        else:
            arg_t.extent.extents = [*local_extent]

        arg_t.extent.sort_extents()

        #print(f"Extent for {arg.name}: {arg_t.extent.extents}")

    # For each output, the extent is the union of the uses
    for output, output_t in zip(node.outputs, node.operation_type.destination):
        output_t.extent.extents = use_offsets
        output_t.extent.sort_extents()

def _walk_IfBlock(uses: dict[sast.Identifier, list[def_use_analysis.ScopedUse]],
                  computation: sast.ComputationBlock,
                  node: sast.IfBlock):
    assert len(node.outputs) == 1, "Only a single output is supported for now"
    # The if-block takes the predicate as input and outputs a field
    use_offsets = _offsets_of_uses_in_scope(uses, computation, node.outputs[0])

    # The output type is the union of the uses
    # For each output, the extent is the union of the uses
    for output, output_t in zip(node.outputs, node.operation_type.destination):
        output_t.extent.extents = use_offsets
        output_t.extent.sort_extents()

    # The input type is [0, 0, 0]
    node.operation_type.source[0].extent.extents[:] = [sast.Offset((0, 0, 0))]


def _walk_ReturnOp(uses: dict[sast.Identifier, list[def_use_analysis.ScopedUse]],
                   computation: sast.ComputationBlock,
                   node: sast.ReturnOp):
    # TODO: Implement?
    pass



def _offsets_of_uses_in_scope(uses: dict[sast.Identifier, list[def_use_analysis.ScopedUse]],
                              computation: sast.ComputationBlock,
                              id: sast.Identifier) -> list[sast.Offset]:
    """
    Get the offsets of the uses of a field in the current scope.
    :param uses: The uses dictionary
    :param computation: The current computation block
    :param id: The identifier
    :return: A list of offsets
    """
    uses_of_result = uses.get(id) or []
    uses_of_result = [u for u in uses_of_result if u.definition_scope == computation]
    # Concatenate all the extents from uses_of_result
    use_offsets = []
    for use in uses_of_result:
        #print(f"Use of {id.name} with extent {use.field_type.extent.extents}")
        use_offsets.extend(use.field_type.extent.extents)
    if len(use_offsets) == 0:
        print(f"WARNING: No uses of {id.name} found within current scope.")

    # Remove any unknown offsets: THIS SHOULD NOT HAPPEN!
    use_offsets = [o for o in use_offsets if all(i != "?" for i in o)]
    #assert all(o[i] != "?" for o in use_offsets for i in range(3)), f"Offset of uses of {id.name} must be known before inference"
    return use_offsets


def _minkowski_sum(a: Collection[sast.Offset],
                   b: Collection[sast.Offset]) -> set[sast.Offset]:
    """
    Compute the minkowski sum of two sets of extents
    :param a:
    :param b:
    :return:
    """
    result = set()
    for a_extent in a:
        for b_extent in b:
            result.add(a_extent + b_extent)
    return result

