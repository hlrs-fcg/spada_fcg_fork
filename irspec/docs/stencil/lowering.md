# Lowering to Spatial IR


This document describes the lowering of the stencil IR to the Spatial IR.

Assuming we have decided on the placement of fields, we can now lower the stencil IR to the Spatial IR.

On a high level, the lowering proceeds hierarchically over the stencil IR.

* Each spst.program maps to a kernel in Spatial IR.
* Each spst.computation maps to a phase in Spatial IR.
* Each spst.statement maps to a sequence of operations inside a compute block in Spatial IR.

At the global scope, the lowering places the input and output fields on the PEs.
Moreover, it places fields for intermediate results of the computations on the PEs.

Within each phase, the lowering contains the following steps:

1. **Lowering Memory Placement**: Generate the `place` blocks using the field placement information.
2. **Lowering Communication**: Extract the streams from the stencil IR.
3. **Lowering computations**: Lower the computations to Spatial IR.

## Lowering spst.program

An spst.program is lowered to a Spatial IR kernel.

### Determining the Parameters

The grid size is determined by the domain type, extent type of the input fields,
and the placement strategy of the fields.

We pass the domain and halo sizes of the input fields to the kernel.

[TODO details]

[TODO example]

### Determining the Arguments

Each input field is passed as readonly arguments to the kernel.
Each output field is passed as a mutable arguments to the kernel.

Field types are passed as stream array arguments to the kernel.
Their dimensions are determined by the size of the input fields.

Scalar types are passed as scalar arguments.

## Lowering spst.compute

As a preprocessing step, replace all non-materialized field accesses with the
equivalent accesses to non-intermediate fields.
This ensures that we can lower statements in isolation.

### Lowering Subgrid Expressions

Each partition of fields is placed on a subset of PEs expressed as a [subgrid expression](../../spatial/spatial#subgrid-expressions).

We determine the ranges for the subgrid expressions based on the domain and halo sizes of the input fields.

[TODO details]

### Lowering Memory Placement

We generate one or more `place` block for each partition of fields.

In this `place` block, we place the fields of the partition on the PEs.

### Lowering Communication

The goal of this stage is to associate each non-local field access with a stream.

A field access `f[x, y, z]` is local w.r.t. field `g` if the stride of `f` and `g` are the same
and the offset of `f`, `g`, and the access work out to `(0, 0)` (TODO insert distance formula)

If a field access is not local, we generate a unique stream for it in the dataflow block(s) for its partition.

[TODO details]

### Lowering Computations


#### Parallel Schedule

In a parallel schedule, each operation is lowered one after the other.


#### Statements

A statement may access one or more fields, which may be local to the PE or might require communication.
If communication is needed, this translates into sends/receives in the Spatial IR.
Otherwise, the computation is performed locally on the PE using a map.


???+ example "Lowering a Statement"

    Assume we have the following statement in the stencil IR:
    
    ```mlir
    %out_1 = spst.statement(%in) : spst.field<spst.cartesian<?,?,?>, spst.extent<(?, ?, ?)> -> spst.field<spst.cartesian<?,?,?>, spst.extent<(?, ?, ?)> {
            spst.return ((-4.0 * %in[0, 0, 0]) - %in[-1, 0, 0]) + %in[1, 0, 0])
        }
    ```

    and that the fields `in` and `out_1` are placed with offset (0, 0) and stride (1, 1) on the PE.
    We already have communication streams `in_stream_0` and `in_stream_1` for the field access
    `in[-1, 0, 0]` and `in[1, 0, 0]`.

    Then, all non-(0, 0, x) accesses are translated into sends/receives, while the local computation is translated into a map.
    
    The Spatial IR for this statement would look like this:
    
    ```rust
    // Lowered Spatial IR

    // Translate local computation -4.0 * %in[0, 0, 0] into a map
    map for k in [0:K] {
        out_1[k] = -4.0 * in[k];
    }

    // Send, receive, for %in[-1, 0, 0]
    completion c1 = send(in, in_stream_0);
    await foreach k, x in [0:K, receive(in_stream_0)] {
        out_1[k] = out_1[k] - x;
    }
    await c1;

    // Send, receive, for %in[1, 0, 0]
    completion c2 = send(in, in_stream_1);
    await foreach k, x in [0:K, receive(in_stream_1)] {
        out_1[k] = out_1[k] + x;
    }
    await c2;

    ```


#### Forward/Backward Schedule

[This translates into a for-loop in the Spatial IR.]

A computation with forward or backward schedule is lowered by generating a for-loop in the Spatial IR,
looping over the z-dimension of the field in ascending or descending order, respectively.

The computation may contain some accesses to read-only fields.
These fields are communicated and stored locally at the beginning of the loop.

Then, we enter the main for-loop.
Whenever we encouter a non-local field access we send a single value to the corresponding stream.
Then, we receive the value and store it in a local variable or directly use it in the computation.
Finally, we await the completion of the communication.

[TODO details & example]
