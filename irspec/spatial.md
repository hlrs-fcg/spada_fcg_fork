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

#### Constant Literals

We may use constant literals to represent constant compile-time values. 

For example `0`, `1`, `1024`, `-12` are constant literals. 
Constant integer literals are `i64` and constant floating-point literals are `f32`. 


#### Channels

A channel corresponds to a virtual communication channel between PEs or the host device and the PEs.

For any scalar type `T`,  `channel<T>` indicates the corresponding element type sent over the channel.

Channels do not send a predetermined number of elements, but the sender and receiver must agree on the number of elements sent and received.
This can be done explicitly (when the size is known from the parameters) or implicitly (by sending a completion signal with/after the last element).

#### Arrays

Any scalar or channel type `T` and one or more parameter expressions `S_1`, `S_2`, ... `S_d` may be used to create an array type `T[S_1, S_2, ... S_d]`.
It represents a d-dimensional array of type `T`, where the i-th dimension contains `S_i` elements.

For example, `f32[10]`, `i32[I+2, J+2]` indicate array types.

#### Parameters

Parameter literals are placeholders for an actual value that will be substituted with an integer 
value at compile time. They are denoted by capital letters or capital letters followed by a number string.
For example, `I`, `J`, `K`, `I001` denote parameters.

#### Variables

A variable starts with a lower case letter and may contain letters, numbers, and underscores.
For example, `x`, `y`, `my_variable`, `my_Variable_2` are valid variable names.

```
variable ::= [a-z][a-zA-Z0-9_]*
```

A variable is in scope if it is declared in the current block or any enclosing block.

#### Parameter Expressions

A parameter expression is an expression that may depend on parameters and constant integer literals. 

```
parameter_expression ::= constant_literal | parameter_literal | parameter_expression + parameter_expression | parameter_expression - parameter_expression | parameter_expression * parameter_expression | parameter_expression // parameter_expression | parameter_expression % parameter_expression | (parameter_expression)
```
where // denotes integer division and % denotes modulo.

For example, `I`, `J+2`, `10`, `(I+J) // 2` are parameter expressions.

#### Expressions

An expression may depend on parameters, constants, and in-scope variables.

```
array_expression ::= variable[int_expression]
int_expression ::= constant_literal | parameter_literal | variable | expression + expression | expression - expression | expression * expression | expression // expression | expression % expression | (expression)
expression ::= constant_literal | parameter_literal | variable | array_expression | expression + expression | expression - expression | expression * expression | expression / expression | expression // expression | expression % expression | (expression) 
```
where // denotes integer division and % denotes modulo.

For example, `I`, `J+2`, `i`, `I+i` are integer expressions and `a[k+1]` is an array expression.

`int` expressions must be of type `i64` and `array` expressions must be of type `T[S_1, S_2, ... S_d]`
for some scalar type `T` and parameter expressions `S_1`, `S_2`, ... `S_d`.

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

```rust
kernel kernel_name<parameters> (arguments) {
  // Kernel definition
}
```
where parameters is a list of parameter literals, and arguments is a list of arguments to the kernel.

#### Arguments

An argument is a named and typed channel array or scalar variable that is passed to a kernel.
```
argument ::= T variable_name | T readonly variable_name | T writeonly variable_name | T compiletime variable_name
```
where `T` is a scalar type, channel type, or channel array type and `variable_name` is a variable name.
Notable, it is not possible to pass scalar arrays as arguments, instead,
arrays must be read through channels.
[TODO: Discuss the rationale for this, we can always pass a channel array and read the scalar array from it.]

If an argument may be *only* read from or written to, it is marked as `readonly` or `writeonly`, respectively.

For example, `channel<f32>[I, J] readonly arg1`, `channel<f32>[I, J] writeonly arg2`, `f32 arg3` are arguments.

If an argument is known at compiletime it may be annotated with `compiletime`.
It must be provided at compilation time together with the parameters.

#### Kernel semantics

A kernel gets the memory of its arguments from a host device or other kernel,
runs the computation, and returns the results to the host device.
This is done through communication channels, which are explicitly defined in the kernel arguments.
Inputs and outputs may be sent and received in a streaming fashion.
If an argument channel may be only read from or written to, it is marked as `readonly` or `writeonly`, respectively.

### Place block

All array data is placed on the PEs using one or more `place` blocks.
A `place` block is used to describe the placement of data on the PEs.

The syntax of the place block is as follows:
```rust
place var_1, var_2 in subgrid_expression {
   // Statements
}
```
Where `var_1`, `var_2` are variables that are bound to the coordinates of the PEs in the subgrid.

For example:
```rust
place i, j in [0:I:2, 0:J] {
    // Statements
}
```

Semantically, the place block iterates over every value in subgrid_expression and 
allocates the memory as described in the block. The variables become bound to the coordinates of the PEs in the subgrid.
They can be used by statements in the place block.

Within the place block, the following statement is supported:

- Allocate a local array: `T[S_0, ...] local_name;`

#### Restrictions

The subgrid of the `place` block is given by the PEs that lie in the `subgrid_expression`. An array may be placed using multiple `place` blocks.
However, each `local_name` may appear at most once for any given PE over all `place` blocks.

### Dataflow block

All communication is set up in one or more `dataflow` blocks, which describe the communication channels between PEs.

The syntax of the dataflow block is as follows:
```rust
dataflow variables in subgrid_expression {
  //Statements
}
```
The subgrid of the dataflow block is given by the PEs
that lie in the `subgrid_expression`.
The subgrids of the dataflow blocks must be disjoint.

The dataflow block can be set up to support various types of channels.
Currently, only *relative channel* indexing is supported:

#### Relative Channel Declaration

Inside a `dataflow block`, a relative communication channel is declared as follows:
```
channel<T> channel_name = relative_channel(dx, dy);
```
where `T` is a scalar type and `dx` and `dy` are parameter expressions that describe the relative position of the target PE.
This describes a streaming communication channel for sending from the current PE at some
position `(i ,j)` to the PE at the relative position `(i+dx, j+dy)`, and simultaneously
a channel for receiving from the PE at the relative position `(i-dx, j-dy)` at the current PE at `(i, j)`.

For example,
```rust
dataflow i, j in [0:I, 0:J] {
    channel<f32> eastwards = relative_channel(1, 0);
    channel<f32> westwards = relative_channel(-1, 0);
    channel<f32> northwards = relative_channel(0, -1);
    channel<f32> southwards = relative_channel(0, 1);
}
```
describes four communication channels to the east, west, north, and south of each PE.

For example,
```
dataflow i, j in [0:I, 0:J] {
    channel<f32> two_north = relative_channel(0, -2);
}
```
describes a communication channel that sends `f32` data two PEs to the north. 

Note that the channel declaration does not imply that any data is ever sent over the channel.
It merely declares the existence of a virtual communication channel.

### Task block

The computation is described in one or more `task` blocks.
Computation is inherently triggered by receiving data from a stream.
Computations may return completions that may trigger other tasks.


The task block is defined as follows:
```rust
task variables in subgrid_expression {
  // Statements
}
```

The subgrid of the task block is given by the PEs
that lie in the `subgrid_expression`.

Task blocks may contain the following statements, some of which are
asynchronous and return completions that may be used to synchronize tasks.
```rust
// Send (asynchronous)
completion_name = send(local_array, channel_name);
// After completion
after (completion_name) {
  // Statements
}
// Foreach loop over a receive() stream until the sender is done (asynchronous)
completion completion_name = foreach iteration_variable_name in [receive(channel_name)] {
  // Assignment statements
}

// Foreach loop over a receive() stream of defined size (asynchronous)
completion completion_name = foreach iteration_variable_names, data_variable_name in [parameter_expressions, receive(channel_name)] {
  // Assignment statements
}
// Parallel map (asynchronous)
completion completion_name = map variable_names in [range_expression] {
  // Assignment statements
}
// Sequential for loop
for variable_name in [range_expression] {
  // Assignment statements or nested for-loops
}
// Asynchronous block (asynchronous)
completion completion_name = async {
  // Statements
}
```
An assignment statement is of the form 
```rust
array_expression = expression;
// or
variable = expression;
```


#### Streaming Data with `send`

Inside a `task` block, the `send` statement sends data asynchronously through a `channel`.

```
completion completion_name = send(local_array, channel_name);
```

The `local_array` must be allocated for each PE in the subgrid
in some `place` block. Similarly, the `channel_name` must be declared in a `dataflow` block
for each PE in the subgrid.

The `completion_name` is a completion handle that may be used to wait for the completion of the send task.
Note that the completion is triggered when the data has been sent, not when it is received.
The completion merely indicates that the data in `local_array` may be safely overwritten
without affecting the result of the computation.

*Data Races*. Performing multiple sends to the same channel concurrently is considered a data race on the channel.
You must synchronize the sends using completions. Two sends are considered concurrent if they are not ordered by `after`.

#### Receiving Streaming Data with `receive`

Inside a `task` block, the `receive` operation wraps a channel to receive a stream of data from it.

```rust
receive(channel_name)
```

Send and receive calls must be compatible with the definitions of the channels in the dataflow blocks
and must be matched across PEs. In particular, if there is a send from PE `A` to PE `B`, there must be one or more corresponding receives from PE `B` to PE `A`.
Similarly, if there is a receive at PE `B`, there must be one or more corresponding sends with destination `B`.
Such a pair of matched send and receive's for a channel is called a *stream edge* from `A` to `B`.

Note that a `receive` operation does not imply that any data is actually received,
it merely declares the existence of a stream edge.

*Deadlocks*. Failure to construct proper stream edges may result in a *deadlock*. The compiler
will check these constraints and report potential deadlocks on a best-effort basis.

*Data Races*.
Two receives in the same task block are considered concurrent if they are not ordered by `after`.
Receiving from the same channel multiple times concurrently is considered a data race on the channel.

#### Managing Concurrency with `after`

Inside a `task` block, the `after` statement is used to trigger a computation after a completion has been received,
and introduce ordering constraints between tasks. These can be used to avoid data races on strams
and arrays.

```rust
after (completion_name) {
  // Statements
}
```

The statements within the `after` block are executed after the completion `completion_name` has triggered.

For example, the following code sends data to `channel_1`
and then sends data to `channel_2` after the completion of the first send and after rewriting the data array.
```rust
completion comp_1 = send(local_array, channel_name);
after (comp_1) {
    after (comp_2) {
        send(local_array, channel_name);
    }
}
```


#### Processing Data Streams with `foreach`

Inside a `task` block, a `foreach` loop can be used to apply a computation to a stream of data.
For each element in the stream, the computation is executed.
The elements are processed in the order they are received.

One may either provide the number of elements to receive, or receive until the sender is done.
```rust
// Receive until the sender is done
completion completion_name = foreach iteration_variable_name in [receive(channel_name)] {
  // Assignment statements
}

// Receive a fixed number of elements
completion completion_name = foreach iteration_variable_names, data_variable_name in [parameter_expressions, receive(channel_name)] {
  // Assignment statements
}
```
One may specify multiple iteration variables, but only a single data variable in the `foreach` loop.
The data variable is bound to the received data.
The iteration variables are bound to the indices of the received data, which is
interpreted as a multi-dimensional array in *row-major* order.

If the number of elements received is known, it is preferable to specify it explicitly in order
to allow for performance optimizations.

For example, the following code receives data from `channel_1` for `K` elements
and assigns the received data to the array `a`.
```
completion completion_name = foreach k, x in [0:K, receive(channel_1)] {
    a[k] = x;
}
```

The `completion_name` is a completion handle that may be used to wait for the completion of the task.
Note that the completion is triggered when the data has been received, not when it is sent.
After the completion triggers, the channel may be used for other sends or receives.

*Deadlocks*:
The sizes sent and received must match:

* This means that for each `send` statement on a given channel, there can be at most one `foreach` loop that does not specify the number of elements to receive.

* If there are multiple `foreach` loops iterating over the same channel that *do* specify the number of elements to receive,
the total sizes must match the total sizes of the arrays that are sent through the channel.

*Failure to correctly match the sizes sent and received may result in a deadlock.*


#### Processing arrays in parallel with `map`

Inside a `task` block, the `map` statement is used to apply a computation to each element of an array.

```rust
completion comp = map variable_names in [range_expression] {
  // Assignment statements
}
```
There is no guarantee on the order in which the map is executed.
Therefore, the map must not contain loop-carried dependencies.

#### Processing arrays sequentially with `for`

Inside a `task` block, the `for` statement is used to apply a computation to each element of an array in a sequential order.

```rust
for variable_name in [range_expression] {
  // Assignment statements or nested for-loops
}
```

A `for` loop does not return completions, as it executes sequentially
and in-order.

#### Computing asynchronously with `async`

Inside a `task` block, an `async` block is used to execute a computation asynchronously.

```rust
completion comp = async {
  // Assignment statements or nested for-loops
}
```

#### Await completions with `await`

[This is an alternative to `after`, should we keep it? It avoids the need to nest `after` statements.]
Inside a `task` block, an `await` statement is used to wait for a completion to trigger.
The `await` can be immediately applied to an asynchronous operation or a completion name.
```rust
await statatement;
await completion_name;
```

For example,
```rust
// Execute a map and wait for its completion
await map i in [0:10] {
    // Statements
}
// Wait for completion of a send
await send(local_array, channel_name);
// Wait for completion of a receive
await foreach k, x in [0:K, receive(channel_name)] {
  // Statements
}
// Wait for a completion
await comp;
```
Semantically, it equivalent to applying an `after` statement with the completion as the argument
immediately after the statement and then executing the remaining statements inside the `after` block.

For example:
```rust
await map i in [0:10] {
    // Statements
}
for i in [0:10] {
    // Statements
}
// Is equivalent to:
completion comp = map i in [0:10] {
    // Statements
}
after (comp) {
    for i in [0:10] {
        // Statements
    }
}
```

Note that statements inside an `await` may still be pre-empted by other asynchronous operations!

[Developer Note: Having statements with immediate awaits means we can put them on the 'main' thread/task, which
might save task IDs and avoid overhead of creating/launching a new thread/task.]

#### Semantics of Asynchronous Statements

Asynchronous statements may, but do not necessarily run in parallel.
Instead, they may execute in any order and may be interleaved with other statements.
An asynchronous statement may be pre-empted at any time, even partially during its execution.
Hence, it is imperative to avoid **data races**:

Within a task block, a statement inside a `foreach`, `async`, or `map` is considered concurrent with another statement if they are not ordered by an `after` statement.
Writing to an array in a statement of a `foreach`, `async`, or `map` block while concurrently reading from it 
or writing to it in another statement anywhere is considered a *data race*
and is considered undefined behavior. 
In particular, sending data from an array while concurrently
writing to it is considered a data race. Also, writing to the same
array twice inside the same `foreach`, `async`, or `map` block is
considered a data race. 
You must synchronize such statements using the completions.
The motivation for this strict definition is
to allow for parallelization, vectorization, and reordering of statements.

Some asynchronous statements cannot make progress until some event occurs.
It is guaranteed that if there exists a statement that
can make progress, at least one of them will make progress.
There is no guarantee of fairness. 
Failure to guarantee completion regardless of progress order constitutes a **deadlock**.

For example, each iteration of a `foreach` waits for the receival of data.
An `after` statement waits for the completion of a task.
A `send` statement requires that the channel has space to carry the data
and may stall if the receiver is not ready to receive the data.
A deadlock-free program will ensure that all PEs can make progress
eventually.

## Phases

One may define multiple phases in a kernel.
Each phase may contain one or more `place`, `dataflow`, and `task` blocks.


For a `place` block defined in the outermost scope, the variables defined therein are in-scope for all `task` blocks in all phases.
For a `place` block within a phase, the variables defined therein are in-scope for the `task` blocks in that phase.

Similarly, for a `dataflow` block defined in the outermost scope, the channels defined therein are in-scope for all `task` blocks in all phases.
For a `dataflow` block within a phase, the channels defined therein are in-scope for the `task` blocks in that phase.

After each `task` block, there is an implicit barrier that waits for all completions to be triggered before starting the next task.
Note that this does *not* imply that all PEs have executed the task.
Within each phase, there can be at most one task block defined per PE.

If one or more phases are defined, no task block may be defined in the outermost scope.
If no phases are explicitly defined, the task blocks are implicitly put in a single phase.
For each phase, there can be at most one task defined per PE.
If multiple tasks are defined per PE per phase, the behavior is undefined.

Phases run in the order they are defined in the code.

For example:
```rust
place for i, j in [0:I, 0:J] {
    f32[K] a <- arg1[i, j, 0:K];
}

dataflow for i, j in [0:I, 0:J] {
  channel<f32> output = argument[i, j];
}

phase {
  place for i, j in [0:I, 0:J] {
    f32[K] b <- arg2[i, j, 0:K];
  }
   
  dataflow for i, j in [0:I, 0:J] {
    channel<f32> eastwards = relative_channel(1, 0);
  }
  
  task for i, j in [0:I, 0:J] {
     // Within this task block:
     // b and a are in scope, eastwards is in scope, and output is in scope
  }

}

phase {

  place for i, j in [1:I-1, 1:J-1] {
    f32[K] c <- arg3[i, j, 0:K];
  }

  dataflow for i, j in [1:I-1, 1:J-1] {
    channel<f32> westwards = relative_channel(-1, 0);
  }
  
  task for i, j in [1:I-1, 1:J-1] {
    // Within this task block:
    // c is in scope, westwards, and output is in scope
  }

}
```



## Examples

### Vertical Advection


```rust
kernel vadv<I,J,K>(channel<f32>[I, J] utens_stage,
                  channel<f32>[I, J] readonly u_stage,
                  channel<f32>[I, J] readonly wcon,
                  channel<f32>[I, J] readonly u_pos,
                  channel<f32>[I, J] readonly utens,
                  channel<f32>[I, J] writeonly datacol,       
                  f32 compiletime dtr_stage,
                  f32 compiletime bet_m,
                  f32 compiletime bet_p) {
  
  ////
  // Read Inputs

  // A place block on the outer scope carries data between the input phase
  // and the computation phase.
  place i, j in [0:I, 0:J] {
      # Inputs & Outputs arrays
      f32[K] utens_stage_l;
      f32[K] utens_stage_l;
      f32[K] u_stage_l;
      f32[K] wcon_l;
      f32[K] u_pos_l;
      f32[K] utens_l;
      f32[K] datacol_l;
  }
  
  phase {

    // Set up communication channels
    // The only PE-PE communication is for wcon, which is sent to the west  
    // We additionally set up a channel for the inputs and outputs
    dataflow i, j in [0:I, 0:J] {
      channel<f32> u_stage_c = u_stage[i, j];
      channel<f32> wcon_c = wcon[i, j];
      channel<f32> u_pos_c = u_pos[i, j];
      channel<f32> utens_c = utens[i, j];
      channel<f32> utens_stage_c = utens_stage[i, j];
    }
    
    // Copy the data
    
    task i, j in [0:I, 0:J] {
        foreach k, x in [0:K, receive(u_stage_c)] {
            u_stage_l[k] = x;
        }
        foreach k, x in [0:K, receive(wcon_c)] {
            wcon_l[k] = x;
        }
        foreach k, x in [0:K, receive(u_pos_c)] {
            u_pos_l[k] = x;
        }
        foreach k, x in [0:K, receive(utens_c)] {
            utens_l[k] = x;
        }
        foreach k, x in [0:K, receive(utens_stage_c)] {
            utens_stage_l[k] = x;
        }
        // Exploits implicit completions at the end of each phase
    }
  }
  
  ////
  // Computation

  phase {
  
    place i, j in [0:I, 0:J] {
      # Local variables live inside the phases scope
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
  
    // Set up communication channels
    // The only PE-PE communication is for wcon, which is sent to the west  
    // We additionally set up a channel for the outputs
    dataflow i, j in [0:I, 0:J] {
      channel<f32> westwards = relative_channel(-1, 0);
      
      channel<f32> utens_stage_c = utens_stage[i, j];
      channel<f32> datacol_c = datacol[i, j];
    }
  
    ////
    // Computation tasks
  
    task i, j in [0, 0:J] {
      // Boundary condition
      // ...
      foreach k, x in [0:K, receive(westwards)] {
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
              gav[k] = -0.25 * x * wcon_l[k];
              gcv[k-1] = 0.25 * x * wcon_l[k];
          }
     
          after (wcon_interval_2) {
              // Rest of the forward pass
              // Cannot be in the foreach block because of a data race on gav and gcv
              for k in [1:K] {
                as_[k] = gav[k] * bet_m;
                cs[k] = gcv[k] * bet_m;
                acol[k] = gav[k] * bet_p;
                ccol[k] = gcv[k] * bet_p;
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
              for k in [K-2:0:-1] {
                datacol_l[k] = dcol_2[k] - ccol_2[k] * datacol_l[k+1];
                utens_stage_l[k] = dtr_stage * (datacol_l[k] - u_pos_l[k]);
              }
  
              // Copy the data to the output
              send(datacol_l, datacol_c);
              send(utens_stage_l, utens_stage_c);
          }
        }   
    }
  }

}


```


### 2D Laplacian

```rust
kernel laplacian<I,J,K> (channel<f32>[I+2, J+2] readonly in_field,
                         channel<f32>[I, J] writeonly lap_field) {
    
  ////////////////////////////////
  // Data placement
  

  
  place i, j in [0:I+2, 0:J+2] {
      f32[K] local_input;
  }
  
  place i, j in [1:I+1, 1:J+1] {
      f32[K] local_result;
  }
  
  /// Copy input data
  
  phase {
  
    dataflow i, j in [0:I+2, 0:J+2] {
      channel<f32> in_field_l = in_field[i, j];
    }
  
    task i, j in [0:I+2, 0:J+2] {
      foreach k, x in [0:K, receive(in_field_l)] {
        local_input[k] = x;
      }
    }
  }
  

  phase {
    // Set up communication channels
    dataflow i, j in [0:I+2, 0:J+2] {
       channel<f32> eastwards = relative_channel(+1, 0);
       channel<f32> westwards = relative_channel(-1, 0);
       channel<f32> northwards = relative_channel(0, -1);
       channel<f32> southwards = relative_channel(0, +1);
       
    }
    
    dataflow i, i in [1:I+1, 1:J+1] {
        channel<f32> lap_field_l = lap_field[i, j];
    }
  
    // Edge senders
    task i, j in [0, 1:J] {
        // Streaming send to the right
        send(local_input, eastwards);
        // We receive nothing
    }
    
    task i, j in [I+1, 1:J] {
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
  
        // No data race, both map and send are reading from local_input
        send(local_input, westwards);
        after (f) {
          // Writing to local_result form the map would be considered a data race.
          completion w = foreach k, x in [0:K, receive(westwards)] {
              local_result[k] -= x;
          }
          // Writing to the same array from multiple foreach blocks concurrently
          // is considered a data race.
          // Hence, we need to run one after the other.
          after (w) {
            send(local_input, eastwards);
            completion e = foreach k, x in [0:K, receive(eastwards)] {
                local_result[k] -= x;
            }
            after (e) {
              // ...
              
              // after all computation has finished:
              send(local_result, lap_field_l);   
            }
          }
  
        }
    }  
  }
}
```

### Streaming 1D Convolution

Performs the convolution of a 1D kernel with a streaming K-D input array that changes over time.
That is in each time step, we receive an array of K elements, and we convolve it with the kernel.

The kernel is of size 3.
While the data is being streamed, it is convolved with the kernel and streamed to the output.

```rust
kernel conv<J>(channel<f32>[J] readonly input,
               channel<f32>[J] writeonly output,
               f32[3] readonly KERNEL) {

    // Data placement
    place i, 0 in [0:J, 0] {
        f32 y;
        f32[3] kernel <- KERNEL;
    }

    // Communication
    dataflow i, j in [0:J, 0] {
        channel<f32> eastwards = relative_channel(1, 0);
        channel<f32> westwards = relative_channel(-1, 0);
        channel<f32> input_local <- input[i];
        channel<f32> output_local -> output[i];
    }

    // Computation
    task i, j in [1:J-1, 0] {

        // Streaming receive
        // Each PE receives a single scalar per time step
        foreach x in receive(input_local) {

            // Send the data to the right
            send(x, eastwards);
            // Send the data to the left
            send(x, westwards);

            y = x * kernel[1];

            completion east = foreach x, y in [0:1, receive(eastwards)] {
                y = y + x * kernel[0];
            }

            after (east) {
                completion west = foreach x, y in [0:1, receive(westwards)] {
                    y = y + x * kernel[2];
                }

                after (west) {
                    // Send the result to the output
                    send(y, output_local);
                }
            }
        }
    }

    // Left corner
    task i, 0 in [0, 0] {
        // Streaming receive
        foreach x in receive(input_local) {
            // Send the data to the right
            send(x, eastwards);
            
            y = x * kernel[1];

            completion west = foreach x, y in [0:1, receive(westwards)] {
                y = y + x * kernel[2];
            }
                
            after (west) {
                // Send the result to the output
                send(y, output_local);
            }
        }
    }

    // Right corner
    // ...

}
```