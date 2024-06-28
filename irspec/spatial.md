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



### Syntax Fundamentals

#### Scalars

`f16`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `bool` indicate scalar types.

#### Constant literals

We may use constant literals to represent constant compile-time values. 

For example `0`, `1`, `1024`, `-12` are constant literals. 
Constant integer literals are `i64` and constant floating-point literals are `f32`. 

#### Array types

Any scalar type `T` and one or more parameter expressions `S_1`, `S_2`, ... `S_d` may be used to create an array type `T[S_1, S_2, ... S_d]`.
It represents a d-dimensional array of type `T`, where the i-th dimension contains `S_i` elements.

For example, `f32[10]`, `i32[I+2, J+2]` are array types. 

#### Parameters

Parameter literals are placeholders for an actual value that will be substituted with an integer 
value at compile time. They are denoted by capital letters or capital letters followed by a number string.
For example, `I`, `J`, `K`, `I001` denote parameter types.

#### Variables

A variable starts with a lower case letter and may contain letters, numbers, and underscores.
For example, `x`, `y`, `my_variable`, `my_variable_2` are valid variable names.

A variable is in scope if it is declared in the current block or any enclosing block.

#### Arguments

An argument is a named and typed array or scalar variable that is passed to a kernel.
```
argument ::= T variable_name | T readonly variable_name | T writeonly variable_name
```
where `T` is a type and `variable_name` is a variable name.

If an argument may be *only* read from or written to, it is marked as `readonly` or `writeonly`, respectively.

For example, `f32[I, J] readonly arg1`, `f32[I, J] writeonly arg2`, `f32[1024] arg3` are arguments.

#### Parameter Expressions

A parameter expression is an expression that may depend on parameters and constant literals. 
 
For example, `I`, `J+2`, `10`, `I+J` are parameter expressions.

#### Expressions

An expression may depend on parameters, constants, in-scope variables, and arguments.

For example, `I`, `J+2`, `i`, `I+i` are integer expressions.

#### Range expressions

A `range_expression` can be constructed using the following syntax:
```
range_expression ::= start:stop | start | start:stop:step
```
where `start`, `stop`, `step` are integer expressions. If all expressions are parameter expressions, the
range expression is a parameter range expression.
The start is inclusive, and the stop is exclusive. The `step` describes the stride of the range.

#### Lists

A list of elements of `X` is separated by commas.

For example, `1, 2, 3` is a list of constant literals.
`x, y, z` is a list of variables.
`f32[10], i32[I+2, J+2]` is a list of array types.
`I+2, J+2` is a list of parameter expressions.

### The coordinate grid

The coordinate grid has two dimensions, `x` and `y`.
The origin `(0, 0)` is at the north-west corner of the grid.
The `x` axis increased towards the east, and the `y` axis increases towards the south.


#### Subgrid expressions

A subgrid expression is given by 
```
subgrid_expression ::= [parameter_expression, parameter_expression]
```
and it describes a subgrid of the PEs.

For example, `[0:I, 0:J]` describes the entire grid of PEs.
`[0:I:2, 0:J/2]` describes every second PE in the `x` direction and the first half of PEs in the `y` direction.

### Kernel

A kernels abstracts a computation that is executed on a grid of processing elements (PEs).
A kernel is defined using the following syntax:

```
kernel kernel_name<parameters> (arguments) {
  // Kernel definition
}
```
where parameters is a list of parameter literals, and arguments is a list of arguments to the kernel.

A kernel gets the memory of its arguments from a host device,
runs the computation, and returns the results to the host device.
The returns values are specified as arguments to the kernel.
Inputs and outputs may be sent and received in a streaming fashion.
If an argument may be only read from or written to, it is marked as `readonly` or `writeonly`, respectively.

### Place block

All array data is placed on the PEs using one or more `place` blocks.
A `place` block is used to describe the placement of data on the PEs.

The syntax of the place block is as follows:
```
place var_1, var_2 in subgrid_expression {
   // Statements
}
```
Where `var_2`, `var_2` are variables that are bound to the coordinates of the PEs in the subgrid.

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

- Allocate a local array: `T[S_0, ...] local_name;`
- Read from an input array as a copy: `T[S_0, ...] local_name <- argument_name[range_expression, ...];`
- Write to an output array as a copy: `T[S_0, ...] local_name -> argument_name[range_expression, ...];`

When using `<-` or `->` the number of dimensions of the local array must match the number of dimensions of size `>1` of the argument array.
Any dimensions of size `1` are squeezed out or unsqueezed unless the number of dimensions match exactly.

For example:
```
place i, j in [0:I, 0:J] {
    f32[K] local_name1 <- arg1[i, j, 0:K];
    f32[K] local_name2 -> arg2[0:K];
    f32[J/2, K] local_name3 <- arg3[0:J:2, 0:K];
    f32[1, 1, K] local_name4 <- arg1[i, j, 0:K];
}
```

In copy mode, inputs are read from before any tasks execute and outputs are copied to the host once all tasks have executed.
(Note that this is a bit restrictive, it does not yet allow streaming of data, and sending back partial results)

#### Restrctions

The subgrid of the `place` block is given by the PEs that lie in the `subgrid_expression`. An array may be placed using multiple `place` blocks.
However, each `local_name` may appear at most once for any given PE over all `place` blocks.

If the same memory location of an output array is written to by multiple PEs,
this constitutes a *race condition* and is undefined behavior. 


### Dataflow block

All communication is set up in one or more `dataflow` blocks, which describe the communication streams between PEs.

The syntax of the dataflow block is as follows:
```
dataflow variables in subgrid_expression {
  //Statements
}
```
The subgrid of the dataflow block is given by the PEs
that lie in the `subgrid_expression`.
The subgrids of the dataflow blocks must be disjoint.

Within the dataflow block, the following statements are supported:

#### Relative Stream Declaration

A relative stream is declared as follows:
```
stream stream_name = relative_stream(dx, dy);
```
where `dx` and `dy` are parameter expressions that describe the relative position of the target PE.
This describes a streaming communication channel from the current PE at some
position `(i, j)` to the PE at the relative position `(i+dx, i+dy)`.

For example,
```
dataflow i, j in [0:I, 0:J] {
    stream eastwards = relative_stream(1, 0);
    stream westwards = relative_stream(-1, 0);
    stream northwards = relative_stream(0, -1);
    stream southwards = relative_stream(0, 1);
}
```
describes the communication streams to the east, west, north, and south of each PE.

For example,
```
dataflow i, j in [0:I, 0:J] {
    stream two_north = relative_stream(0, -2);
}
```
describes a communication stream that sends data two PEs to the north.

Note that the relative stream declaration does not imply that any data is
ever sent over the stream. It merely declares the existence of a virtual communication channel.

### Task block

The computation is described in one or more `task` blocks.
Computation is inherently triggered by receiving data from a stream.
Computations may return completions that may trigger other tasks.


The task block is defined as follows:
```
task variables in subgrid_expression {
  // Statements
}
```

The subgrid of the task block is given by the PEs
that lie in the `subgrid_expression`.

Tasks blocks *must* define **disjoint subgrids**.
Not every PE must lie in a task block.

#### Sending Data Streams with `send`

The `send` statement sends data to a stream.

```
completion completion_name = send(local_array, stream_name);
```

The `local_array` must be allocated for each PE in the subgrid
in some `place` block. Similarly, the `stream_name` must be declared in a `dataflow` block
for each PE in the subgrid.

The `completion_name` is a completion handle that may be used to wait for the completion of the send task.
Note that the completion is triggered when the data has been sent, not when it is received.
The completion merely indicates that the data in `local_array` may be safely overwritten
without affecting the result of the computation.

*Data Races*. Performing multiple sends to the same stream concurrently is considered a data race on the stream.
You must synchronize the sends using completions. Two sends are considered concurrent if they are not ordered by `after`.

#### Receiving Data Streams with `receive`

The `receive` statement wraps a stream to receive data from it.

```
receive(stream_name)
```

Send and receive calls must be compatible with the definitions of the streams in the dataflow blocks
and must be matched across PEs. In particular, if there is a send from PE `A` to PE `B`, there must be one or more corresponding receives from PE `B` to PE `A`.
Similarly, if there is a receive at PE `B`, there must be one or more corresponding sends with destination `B`.
Such a pair of matched send and receive statements for a stream is called a *stream edge* from `A` to `B`.

*Deadlocks*. Failure to construct proper stream edges may result in a *deadlock*. The compiler
will check these constraints and report potential deadlocks on a best-effort basis [Discuss].

*Data Races*.
Receiving from the same stream multiple times concurrently is considered a data race on the stream.

#### Managing Concurrency with `after`

The `after` statement is used to trigger a computation after a completion has been received,
and introduce ordering constraints between tasks. These can be used to avoid data races on strams
and arrays.

```
after (completion_name) {
  // Statements
}
```

The statements within the `after` block are executed after the completion `completion_name` has triggered.

For example, the following code sends data to `stream_1`
and then sends data to `stream_2` after the completion of the first send and after rewriting the data array.
```
completion comp_1 = send(local_array, stream_name);
after (comp_1) {
    after (comp_2) {
        send(local_array, stream_name);
    }
}
```



#### Processing Data Streams with `foreach`

A `foreach` loop can be used to apply a computation to a stream of data.
For each element in the stream, the computation is executed.
The elements are processed in the order they are received.

One may either provide the number of elements to receive[, or receive until the sender is done].
```
// Receive until the sender is done
// Discuss: I am not sure we even want to allow for this!
// I would for now, leave it out
completion completion_name = foreach iteration_variable_name in [receive(stream_name)] {
  // Assignment statements
}

// Receive a fixed number of elements
completion completion_name = foreach iteration_variable_name, data_variable_name in [parameter_expression, receive(stream_name)] {
  // Assignment statements
}
```
If the number of elements received is known, it is always preferable to specify it explicitly in order
to allow for performance optimizations.

For example, the following code receives data from `stream_1` for `K` elements
and assigns the received data to the array `a`.
```
completion completion_name = foreach k, x in [0:K, receive(stream_1)] {
    a[k] = x;
}
```

The `completion_name` is a completion handle that may be used to wait for the completion of the task.

*Deadlocks*:
The sizes sent and received must match:

* This means that for each `send` statement on a given stream, there can be at most one `foreach` loop that does not specify the number of elements to receive.

* If there are multiple `foreach` loops iterating over the same stream that *do* specify the number of elements to receive,
the total sizes must match the total sizes of the arrays that are sent through the stream.

*Failure to correctly match the sizes sent and received may result in a deadlock.*

*Data Races*: Writing to the same array from mutliple streams concurrently is considered a *data race*
and is undefined behavior. You must synchronize the writes using completions. Two streams are considered concurrent if
they are not ordered by `after`.

#### Processing arrays in parallel with `map`

The `map` statement is used to apply a computation to each element of an array.

```
completion comp = map variable_names in [range_expression] {
  // Scalar assignment statements
}
```
There is no guarantee on the order in which the map is executed.
Therefore, the map must not contain loop-carried dependencies.


#### Processing arrays sequentially with `for`

The `for` statement is used to apply a computation to each element of an array in a sequential order.

```
for variable_name in [range_expression] {
  // Scalar assignment statements
}
```

A `for` loop does not return completions, as it executes sequentially
and in-order.

## Examples

### Vertical Advection


```
kernel vadv<I,J,K>(f32[I, J, K] utens_stage,
                  f32[I, J, K] readonly u_stage,
                  f32[I, J, K] readonly wcon,
                  f32[I, J, K] readonly u_pos,
                  f32[I, J, K] readonly utens,
                  f32[I, J, K] writeonly datacol,       
                  f32 readonly dtr_stage,
                  f32 readonly BET_M,
                  f32 readonly BET_P) {
  
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
  
      completion wcon_interval_1 = foreach k, x in [0:1, receive(westwards)] {
          gav[k] = -0.25 * x * wcon_l[k];
          gcv[k] = 0
          // ...
      }
  
      after (wcon_interval_1) {
        completion wcon_interval_2 = foreach k, x in [1:K, receive(westwards)] {
            // TODO: Question: do we allow multiple statements here? Would be nice to have.
            gav[k] = -0.25 * x * wcon_l[k];
            gcv[k-1] = 0.25 * x * wcon_l[k];
        }
   
        after (wcon_interval_2) {
            // Rest of the forward pass
            // Could also be in the foreach loop
            for k in [1:K] {
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

            // Boundary condition (k=K-1) not shown
            for k in [K-1] {
                /// Boundary condition ...
            }
            // Main backwards loop          
            for k in [K-2:0] {
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
        completion w = foreach k, x in [0:K, receive(westwards)] {
            local_result[k] -= x;
        }
        // Writing to the same array from multiple streams concurrently
        // is considered a data race.
        // Hence, we need to run one after the other.
        after (w) {
          send(local_input, eastwards);
          completion e = foreach k, x in [0:K, receive(eastwards)] {
              local_result[k] -= x;
          }
          after (e) {
            // ...         
          }
        }

      }
  }
}
```


