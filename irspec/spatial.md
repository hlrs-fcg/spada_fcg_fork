# Spatial IR - Level 1 - High-level IR for spatial computations

## Design Goals

- Models processing elements (PEs)
  - Represents coordinates of PEs
  - Can represent sub-grids of PEs
- Models data placement on PEs (memory layout)
  - Support for multi-dimensional arrays (of fixed, but arbitrary dimensions)
  - Support for interleaved / strided layouts
  - Support for placing on sub-regions of PEs grid
  - Inputs and outputs are placed onto the PEs explicitly, through layout description
  - Inputs and outputs are *moved* onto the chip implicitly. TODO: Discuss
  - Inputs and outputs may be streamed into and out of the PEs
  - PEs may have local memory arrays
  - Support for parametric array sizes (provided at compile time)
  - Support for constant array sizes (provided at compile time)
- Models communication between PEs explicitly
  - Communication is streaming (pipelined)
  - Supports relative addressing for communication (e.g., send to the PE to the right or twice to the left)
  - Support for collective communication (e.g., broadcast, reduce, all-reduce, scatter, gather, all-gather)
  - Completion of a communication stream may trigger other communication or computation
- Does not encode platform-specific routing details
- Models vectorizable computations
  - Computation is triggered by receiving data
  - Operations can be expressed in a way that naturally allows for vectorization & streaming processing
  - Support for sequential loops
  - When a computational task finished, it may trigger other tasks
- Models data dependencies
  - Data dependencies are explicit
  - Enable reasoning about deadlock-freedom
  - Allow for scheduling and reordering of tasks

## Specification

### Coordinate Grid

The coordinate grid has two dimensions, `x` and `y`.
The origin `(0, 0)` is at the north-west corner of the grid.
The `x` axis increased towards the east, and the `y` axis increases towards the south.



## Examples


### Vertical Advection

### 2D Laplacian

```
////////////////////////////////
// Data placement (I/O)

f32[I+2, J+2, K] input in_field;
f32[I, J, K] output lap_field;

place i, j in [0:I+2, 0:J+2] {
    f32 in_field[i, j, 0:K];
}

place i, j in [1:I+1, 1:J+1] {
    f32 lap_field[i-1, j-1, 0:K];
}


////////////////////////////////
// Computation and communication


// Set up communication streams
// Communication streams as a first-class concept
stream eastwards = relative_stream(1, 0);
stream westwards = relative_stream(-1, 0);
stream northwards = relative_stream(0, -1);
stream southwards = relative_stream(0, 1);

// Edge senders
map spatial i, j in [0, 0:J] {
    // Streaming read from in_field
    f32[K] tosend <- in_field;
    
    // Streaming send to the right
    send(tosend, eastwards);
    // We receive nothing
}

map spatial i, j in [I+1, 0:J] {
    // Streaming read from in_field
    f32[K] tosend <- in_field;
    
    // Streaming send to the left
    send(tosend, westwards);
    // We receive nothing
}
// ...

map spatial i, j in [1:I+1, 1:J+1] {
    f32[K] local_input <- in_field;
    f32[K] local_result;

    // Streaming parallel computation (map)
    completion f = map k in [0:K] {
        local_result[k] = local_input[k] * 4;
    }
    wait(f);

    send(local_input, westwards);
    completion w = on_receive(westwards, K) -> k, x {
        local_result[k] -= x;
    }

    send(local_input, eastwards);
    completion e = on_receive(eastwards, K) -> k, x {
        local_result[k] -= x;
    }
    // ...

    wait_all(w, e, n, s);

    // Streaming write to output field
    local_result -> lap_field;
}
```