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


### Fundamental Concepts

- Scalars: `f16`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `bool` indicate scalar types.

- Constants: A constant is a scalar value that is known at compile time.
  They may be indicated by adding the `constant` keyword after the type.

- Constant literals: We may use constant literals to represent constant values.
  For example `0`, `1`, `1024` are constant literals. 
  Constant integer literals are `i32` and constant floating-point literals are `f32`.

- Array types: Any scalar type `T` and a `size_expression` may be used to create an array type `T[size_expression]`.
  For example, `f32[10]`, `i32[I+2, J+2]` are array types.

- Parameters: Parameter literals are placeholders for an actual value that will be substituted with an integer 
  value. They are denoted by capital letters or capital letters followed by a number string.
  For example, `I`, `J`, `K`, `I0` can be used as parameter types.

- Variables: A variable starts with a lower case letter and may contain letters, numbers, and underscores.
  For example, `x`, `y`, `my_variable`, `my_variable_2` are valid variable names.

- Integer expressions: An integer expression may depend on parameters, constants, and in-scope variables.
  They are used to specify the size of arrays and access them.
  For example, `I`, `J+2`, `10`, `I+J` are integer expressions.

- A `range_expression` can be constructed using the following syntax:
```
range_expression ::= start:stop | start | start:stop:step
```
where `start`, `stop`, `step` are integer expressions.
The start is inclusive, and the stop is exclusive. The `step` describes the stride of the range.

- A `subgrid_expression` can be constructed from one or more `range_expression` using the following syntax:
```
subgrid_expression ::= range_expression, range_expression, ... 
```

- Coordinate grid. The coordinate grid has two dimensions, `x` and `y`.
The origin `(0, 0)` is at the north-west corner of the grid.
The `x` axis increased towards the east, and the `y` axis increases towards the south.

### Kernel

A kernels abstracts a computation that is executed on a grid of processing elements (PEs).
A kernel is defined using the following syntax:

```
kernel kernel_name<parameters> (arguments) {
  // Kernel definition
}
```
where parameters is a list of parameter literals, and arguments is a list of arguments to the kernel.
The arguments are named and typed.

A kernel gets the memory of its arguments from a host device,
runs the computation, and returns the results to the host device.
The returns values are specified as arguments to the kernel.
Inputs and outputs may be sent and received in a streaming fashion.
If an argument may be only read from or written to, it is marked as `readonly` or `writeonly`, respectively.

### Place block

All array data is placed on the PEs using one or more `place` blocks.
A `place` block is used to describe the placement of data on the PEs.

They syntax of the place block is as follows:
```
place variables in [subgrid_expression] {
   // Statements
}
```
Where `variables` is a list of variables that are bound to the coordinates of the PEs in the subgrid.

For example:
```
place i, j in [0:I:2, 0:J] {
    // Statements
}
```

Semantically, the place block iterates over every value in subgrid_expression and 
allocates the memory as described in the block. The variables become bound to the coordinates of the PEs in the subgrid.
They can be used by statements in the place block.

Within the place block, the following statements are supported:

- Allocate a local array: `T[size_expression] local_name;`
- Read from an input array as a copy: `T[size_expression] local_name <- argument_name[subgrid_expression];`
- Write to an output array as a copy: `T[size_expression] local_name -> argument_name[subgrid_expression];`

In copy mode, inputs are read from before any tasks execute and outputs are read copied to the host once all tasks have executed.
(Note that this is a bit restrictive, it does not yet allow streaming of data, and sending back partial results)


The subgrid of the `place` block is given by the PEs that lie in the `subgrid_expression`. An array may be placed using multiple `place` blocks.
However, each `local_name` may appear at most once for any given PE over all `place` blocks.

### Dataflow block

All communication is set up in one or more `dataflow` blocks, which describe the communication streams between PEs.

The syntax of the dataflow block is as follows:
```
dataflow variables in [subgrid_expression] {
  //Statements
}
```
The subgrid of the dataflow block is given by the PEs
that lie in the `subgrid_expression`.
The subgrids of the dataflow blocks must be disjoint.

Within the dataflow block, the following statements are supported:

#### Stream declaration

- Declare a relative stream: `stream stream_name = relative_stream(dx, dy);`
  where `dx` and `dy` are integer expressions that describe the relative position of the target PE.
  This described a streaming communication channel from the current PE at
  position `(i, j)` to the PE at the relative position `(i+dx, i+dy)`.

### Task block

The computation is described in one or more `task` blocks.
Computation is inherently triggered by receiving data from a stream.
Computations return completions that may trigger other tasks.


The task block is defined as follows:
```
task variables in [subgrid_expression] {
  // Statements
}
```

The subgrid of the task block is given by the PEs
that lie in the subgrid_expression.

Tasks blocks must define disjoint subgrids.

#### send

The `send` statement sends data to a stream.

```
send(local_array, stream_name);
```

The `local_array` must be allocated for each PE in the subgrid
in some `place` block. Similarly, the `stream_name` must be declared in a `dataflow` block
for each PE in the subgrid.

#### on_receive

The `on_receive` statement receives data from a stream and triggers a computation for 
each element.

```
completion comp = on_receive(stream_name, size_expression) -> variable_name_1, variable_name_2 {
  // Scalar assignment statements
}
```
The `size_expression` is the number of scalars of the data that is received.
This may be less than the size of the array that is sent.

The `variable_name_1` refers to the index of the received data.
The `variable_name_2` refers to the data that is received.

The `comp` is a completion handle that may be used to wait for the completion of the task.

#### after

The `after` statement is used to trigger a computation after a completion has been received.

```
after (comp) {
  // Statements
}
```

The statements within the `after` block are executed after the completion `comp` has triggered.

#### map

The `map` statement is used to apply a computation to each element of an array.

```
completion comp = map variable_names in [range_expression] {
  // Scalar assignment statements
}
```
There is no guarantee on the order in which the map is executed.
Therefore, the map must not contain loop-carried dependencies.


DRAFT: instead of on-receive:
Maps may be used to process streaming inputs as they arrive.

```
completion comp = map k, x in receive(stream_1, K) {
    // Scalar assignment statements
    a[k] = x;
}
```

#### for

The `for` statement is used to apply a computation to each element of an array in a sequential order.

```
for variable_name in [range_expression] {
  // Scalar assignment statements
}
```

For loops do not return completions, as they execute sequentially
and in order.

## Examples


### Vertical Advection


```
kernel vadv<I,J,K>(f32[I, J, K] utens_stage,
                  f32[I, J, K] readonly u_stage,
                  f32[I, J, K] readonly wcon,
                  f32[I, J, K] readonly u_pos,
                  f32[I, J, K] readonly utens,
                  f32[I, J, K] writeonly datacol,       
                  f32 constant dtr_stage,
                  f32 constant BET_M,
                  f32 constant BET_P) {
  
  ////
  // Data placement (I/O)

  place i, j in [0:I, 0:J] {
      # Inputs & Outputs
      f32[K] utens_stage_l <- utens_stage[i, j, 0:K];
      f32[K] utens_stage_l -> utens_stage[i, j, 0:K];
      f32[K] u_stage_l <- u_stage[i, j, 0:K];
      f32[K] wcon_l <- wcon[i, j, 0:K];
      f32[K] u_pos_l <- u_pos[i, j, 0:K];
      f32[K] utens_l <- utens[i, j, 0:K];
      f32[K] datacol_l -> datacol[i, j, 0:K];
      
      # Local variables
      f32[K] gav;
      f32[K] gcv;
      f32[K] as_;
      f32[K] cs;
      f32[K] acol;
      f32[K] ccol;
      f32[K] bcol;
      f32[K] correction_term;
      f32[K] dcol;
      f32[K] ccol_2;
      f32[K] dcol_2;
      f32[K] datacol_l;
  }

  ////
  // Communication
  
  // Set up communication streams
  // The only communication is for wcon, which is sent to the west  
  dataflow i, j in [0:I, 0:J] {
    stream westwards = relative_stream(-1, 0);
  }

  ////
  // Computation tasks

  task i, j in [0, 0:J] {
    // Boundary condition
    // ...
    on_receive(westwards, K) -> k, x {
      // ...
    }
  }

  task i, j in [1:I, 0:J] {

      
      send(wcon_local, westwards);
  
      completion wcon_comp_1 = on_receive(westwards, 1) -> k, x {
          gav[k] = -0.25 * x * wcon_l[k];
          gcv[k] = 0
      }
  
      after (wcon_comp_1) {
        completion wcon_comp_2 = on_receive(westwards, K-1) -> k, x {
            gav[k] = -0.25 * x * wcon_l[k];
            gcv[k-1] = 0.25 * x * wcon_l[k];
            as_[k] = gav[k] * BET_M;
            cs[k] = gcv[k] * BET_M;
            acol[k] = gav[k] * BET_P;
            ccol[k] = gcv[k] * BET_P;
            bcol[k] = dtr_stage - acol[k] - ccol[k];
  
            correction_term[k] = -as_[k] * (u_stage_l[k-1] - u_stage_l[k]) - cs[k] * (u_stage_l[k+1] - u_stage_l[k]);
            dcol[k] = dtr_stage * u_pos_l[k] + utens_l[k] + utens_stage_l[k] + correction_term[k];
            
            // Thomas forward
            f32 divided = 1.0 / (bcol[k] - ccol[k-1] * acol[k]);
            ccol_2[k] = ccol[k] * divided;
            dcol_2[k] = (dcol[k] - dcol[k-1] * acol[k]) * divided;
        }
   
        after (wcon_comp_2) {
            // Boundary condition (k=K-1) not shown
            // ...
            // Main backwards loop          
            for k in [K-1:0] {
              datacol_l[k] = dcol_2[k] - ccol_2[k] * datacol_l[k+1];
              utens_stage_l[k] = dtr_stage * (datacol_l[k] - u_pos_l[k]);
            }

        }
        
      }   
  }

}


```


### 2D Laplacian

```
kernel laplacian<I,J,K> (f32[I+2, J+2, K] readonly in_field,
                         f32[I, J, K] writeonly lap_field) {
    
  ////////////////////////////////
  // Data placement
  
  place i, j in [0:I+2, 0:J+2] {
      f32[K] local_input <- in_field[i, j, 0:K];
  }
  
  place i, j in [1:I+1, 1:J+1] {
      f32[K] local_result -> lap_field[i-1, j-1, 0:K];
  }
  
  
  ////////////////////////////////
  // Computation and communication
  
  
  // Set up communication streams
  // Communication streams as a first-class concept
  
  dataflow i, j in [0:I+1, 0:J+1] {
     stream eastwards = relative_stream(1, 0);
     stream westwards = relative_stream(-1, 0);
     stream northwards = relative_stream(0, -1);
     stream southwards = relative_stream(0, 1);
  }
  
  // Edge senders
  task i, j in [0, 0:J] {
      // Streaming send to the right
      send(local_input, eastwards);
      // We receive nothing
  }
  
  task i, j in [I+1, 0:J] {
      // Streaming send to the left
      send(tosend, westwards);
      // We receive nothing
  }
  // ...
  
  task i, j in [1:I+1, 1:J+1] {
   
      // Streaming parallel computation (map)
      completion f = map k in [0:K] {
          local_result[k] = local_input[k] * 4;
      }
  
      after (f) {
        send(local_input, westwards);
        completion w = on_receive(westwards, K) -> k, x {
            local_result[k] -= x;
        }
    
        send(local_input, eastwards);
        completion e = on_receive(eastwards, K) -> k, x {
            local_result[k] -= x;
        }
        // ...
        
        after (w, e, n, s) {  
          // Streaming write to output field
          local_result -> lap_field;
        }
      }
  }
}
```


