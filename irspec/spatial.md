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


#### Streams

A stream corresponds to an abstract way to communicate between PEs or the host device and the PEs.

For any scalar type `T`,  `stream<T>` indicates the corresponding element type sent over the stream.

Streams do not send a predetermined number of elements, but the sender and receiver must agree on the number of elements sent and received.
This can be done explicitly (when the size is known from the parameters) or implicitly (by sending a completion signal with/after the last element).

#### Arrays

Any scalar or stream type `T` and one or more parameter expressions `S_1`, `S_2`, ... `S_d` may be used to create an array type `T[S_1, S_2, ... S_d]`.
It represents a d-dimensional array of type `T`, where the i-th dimension contains `S_i` elements.

For example, `f32[10]`, `i32[I+2, J+2]` indicate array types.

#### Parameters

Parameter literals are placeholders for an actual value that will be substituted with an **integer** 
value at compile time. They are denoted by capital letters or capital letters followed by a number string.
For example, `I`, `J`, `K`, `I001` denote parameters.

#### Variables

A variable name starts with a lower case letter and may contain letters, numbers, and underscores.
For example, `x`, `y`, `my_variable`, `my_Variable_2` are valid variable names.

A variable declaration contains a type `T` and a variable name.
```
variable ::= [a-z][a-zA-Z0-9_]*
variable_declaration ::= T variable_name
```

A variable is in scope if it is declared in the current block or any enclosing block.

#### Parameter Expressions

A parameter expression is an expression that may depend on parameters and constant **integer** literals. 

```
parameter_expression ::= constant_literal | parameter_literal | parameter_expression + parameter_expression | parameter_expression - parameter_expression | parameter_expression * parameter_expression | parameter_expression // parameter_expression | parameter_expression % parameter_expression | (parameter_expression)
```
where // denotes integer division and % denotes modulo.

For example, `I`, `J+2`, `10`, `(I+J) // 2` are parameter expressions.

#### Expressions

An expression may depend on parameters, constants, in-scope variables, and fields.

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
`[0:I:2, 0:J//2]` describes every second PE in the `x` direction and the first half of PEs in the `y` direction.

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

An argument is a named and typed stream array or scalar that is passed to a kernel.
```
argument_name ::= [a-z][a-zA-Z0-9_]*
argument ::= T argument_name | T readonly argument_name | T writeonly argument_name | T compiletime argument_name
```
where `T` is a scalar type, stream type, or stream array type.
Notable, it is not possible to pass scalar arrays as arguments, instead,
arrays must be read through streams.
[TODO: Discuss the rationale for this, we can always pass a stream array and read the scalar array from it.]

If an argument may be *only* read from or written to, it is marked as `readonly` or `writeonly`, respectively.

For example, `stream<f32>[I, J] readonly arg1`, `stream<f32>[I, J] writeonly arg2`, `f32 arg3` are arguments.

If an argument is known at compiletime it may be annotated with `compiletime`.
It must be provided at compilation time together with the parameters.

#### Kernel semantics

A kernel gets the memory of its arguments from a host device or other kernel,
runs the computation, and returns the results to the host device.
This is done through communication streams, which are explicitly defined in the kernel arguments.
Inputs and outputs may be sent and received in a streaming fashion.
If an argument stream may be only read from or written to, it is marked as `readonly` or `writeonly`, respectively.

### Place block

All data is placed on the PEs using one or more `place` blocks.

The syntax of a place block is as follows:
```rust
place i32 variable, i32 variable in subgrid_expression {
   // Statements
}
```
For example:
```rust
place i32 i, i32 j in [0:I:2, 0:J] {
    // Statements
}
```
Where `i`, `j` are `i32` variables that are bound to the coordinates of the PEs in the subgrid.

Semantically, a place block iterates over every value in subgrid_expression and 
allocates field memory as described in the block.
The *fields* declared therein become bound to the coordinates of the PEs in the subgrid.
They can be used by statements in a `compute` block.

Field names follow the same syntax as variable names:
```rust
field_name ::= [a-z][a-zA-Z0-9_]*
```

Within a `place` block, the following statements are supported:

- Allocate a local array field: `T[S_0, ...] field_name;`
- Allocate a local scalar field: `T field_name;`


The subgrid of the `place` block is given by the PEs that lie in the `subgrid_expression`.
An array may be placed using multiple `place` blocks.
However, each `field_name` may appear at most once for any given PE over all `place` blocks.

### Dataflow block

All communication is set up in one or more `dataflow` blocks, which describe the communication streams between PEs.

The syntax of the dataflow block is as follows:
```rust
dataflow i32 variable, i32 variable in subgrid_expression {
  //Statements
}
```
The subgrid of the dataflow block is given by the PEs
that lie in the `subgrid_expression`.
The subgrids of the dataflow blocks must be disjoint.

The dataflow block can be set up to support various types of streams.
Currently, only *relative stream* indexing is supported:

#### Relative Stream Declaration

Inside a `dataflow block`, a relative communication stream is declared as follows:
```
stream<T> stream_name = relative_stream(dx, dy);
```
where `T` is a scalar type and `dx` and `dy` are parameter expressions that describe the relative position of the target PE.
This describes a streaming communication stream for sending from the current PE at some
position `(i ,j)` to the PE at the relative position `(i+dx, j+dy)`, and simultaneously
a stream for receiving from the PE at the relative position `(i-dx, j-dy)` at the current PE at `(i, j)`.

For example,
```rust
dataflow i, j in [0:I, 0:J] {
    stream<f32> eastwards = relative_stream(1, 0);
    stream<f32> westwards = relative_stream(-1, 0);
    stream<f32> northwards = relative_stream(0, -1);
    stream<f32> southwards = relative_stream(0, 1);
}
```
describes four communication streams to the east, west, north, and south of each PE.

For example,
```
dataflow i, j in [0:I, 0:J] {
    stream<f32> two_north = relative_stream(0, -2);
}
```
describes a communication stream that sends `f32` data two PEs to the north. 

Note that the stream declaration does not imply that any data is ever sent over the stream.
It merely declares the existence of a virtual communication stream.

#### Routing Declarations

Optionally, a routing declaration may be set up for each stream.
This declaration describes how the data is routed between the PEs.
In particular, for each stream, the configuration may specify the intermediate hops that the data takes.
Moreover, it may specify a `channel`, which is a limited hardware resource
(a virtual or hardware channel) that is used to route the data.

The routing configuration is set up as follows:
```
stream<T> stream_name = relative_stream(dx, dy) {
    // Optional routing declaration
    hops = [(dx_1, dy_1), (dx_2, dy_2), ... , (dx_n, dy_n)];
    channel = channel_id;
};
```
where `hops` is a list of relative hops that the data takes between the sender and receiver.
Each hop is given by a pair of constant literals, the sum of their absolute value must be 1.
The sum of all the hops must be equal to the relative position of the stream.

If two messages (elements of a `send`) are routed through a PE simultaneously,
it must be ensured that they do not share a `channel`.
Note that the start and end PEs also count as hops implicitly.

For example,
```rust
dataflow i, j in [0:I, 0:J] {
    stream<f32> eastwards = relative_stream(1, 0) {
        hops = [(1, 0)];
        channel = 0;
    };
}
```


If no routing declaration is provided, it is up to the compiler to determine the routing.
This is equivalent to setting `hops = auto` and `channel = auto`.
One may also provide `hops` explicitly, but leave `channel = auto`, which allows the compiler to determine the channel.
```rust
// Example use of channel=auto

dataflow i, j in [0:I, 0:J] {
    stream<f32> eastwards = relative_stream(1, 0) {
        hops = auto;
        channel = auto;
    };
}

```
See the [Semantics of Routing Declarations](#semantics-of-routing-declarations) for how the compiler
checks if routing declarations are correct and how it resolves auto-routing.


### Compute block

The computation is described in one or more `compute` blocks.
Computation is inherently asynchronous, triggered by receiving data from streams.
Statements in the `compute` block may return completions that may trigger other statements.

The compute block is defined as follows:
```rust
compute i32 variable, i32 variable in subgrid_expression {
  // Statements
}
```
For example,
```rust
compute i32 i, i32 j in [0:I, 0:J] {
    // Statements
}
```
where `i`, `j` are `i32` variables that are bound to the coordinates of the PEs in the subgrid.

The subgrid of the `compute` block is given by the PEs
that lie in the `subgrid_expression`.

`compute` blocks may contain the following statements, some of which are
asynchronous and return completions that may be used to synchronize computations.
```rust
// Send (asynchronous)
completion_name = send(local_array, stream_name);

// Foreach loop over a receive() stream until the sender is done (asynchronous)
completion completion_name = foreach variables in [receive(stream_name)] {
  // Statements
}

// Foreach loop over a receive() stream of defined size (asynchronous)
completion completion_name = foreach variables in [parameter_expressions, receive(stream_name)] {
  // Statements
}
// Parallel map (asynchronous)
completion completion_name = map variables in [range_expression] {
  // Assignment statements
}
// Sequential for loop
for variables in [range_expression] {
  // Statements
}
// Asynchronous block (asynchronous)
completion completion_name = async {
  // Statements
}

// Await a completion
await completion_name;
```
An assignment statement is of the form 
```rust
array_expression = expression;
// or
field_name = expression;
```

#### Streaming Data with `send`

Inside a `compute` block, the `send` statement sends data asynchronously through a `stream`.

```
// Send the whole array
completion completion_name = send(local_array, stream_name);

// Send part of an array
completion completion_name = send(local_array[parameter_range_expression], stream_name);

```
The `local_array` must be allocated for each PE in the subgrid
in some `place` block. Similarly, the `stream_name` must be declared in a `dataflow` block
for each PE in the subgrid.

The `completion_name` is a completion handle that may be used to wait for the completion of the send operation.
Note that the completion is triggered when the data has been sent, not when it is received.
The completion merely indicates that the data in `local_array` may be safely overwritten
without affecting the result of the computation.




*Data Races*. Performing multiple sends to the same stream concurrently is considered a data race on the stream.
You must synchronize the sends using completions.
Two sends in the same phase are concurrent if they are not ordered by [`await`](#await-completions-with-await).

For example,
```rust
// Send the first half of the array
completion c1 = send(a[0:K//2], stream_name);
await c1;
// Send the rest of the array
completion c2 = send(a[K//2:K], stream_name);
```

#### Receiving Streaming Data with `receive`

Inside a `compute` block, the `receive` operation wraps a stream to receive a stream of data from it.

```rust
receive(stream_name)
```

Send and receive calls must be compatible with the definitions of the streams in the dataflow blocks
and must be matched across PEs. In particular, if there is a `send` from PE `A` to PE `B`, there must be 
one corresponding `receive` from PE `B` to PE `A`.
Similarly, if there is a `receive` at PE `B`, there must be one corresponding `send` with destination `B`.
Such a pair of matched `send` and `receive`'s for a stream is called a *stream edge* from `A` to `B`.
Note that a `receive` operation does not imply that any data is actually received,
it merely declares the existence of a stream edge.

*Deadlocks*. Failure to construct proper stream edges may result in a *deadlock*. The compiler
will check these constraints and report potential deadlocks on a best-effort basis.

*Data Races*.
Two receives in the same `compute` block are considered concurrent if they are not ordered by `await`.
Receiving from the same stream multiple times concurrently is considered a data race on the stream.


#### Processing Data Streams with `foreach`

Inside a `compute` block, a `foreach` loop can be used to apply a computation to a stream of data.
For each element in the stream, the computation is executed.
The elements are processed in the order they are received.

One may either provide the number of elements to receive, or receive until the sender is done.
```rust
// Receive until the sender is done
completion completion_name = foreach variables in [receive(stream_name)] {
  // Assignment statements
}

// Receive a fixed number of elements
completion completion_name = foreach variables in [parameter_rage_expressions, receive(stream_name)] {
  // Assignment statements
}
```
The last variable is the data variable. The data variable is bound to the received data. Its type must match the type of the stream.

The other variables are iteration variables. They must be of type `i32`.
One may specify multiple parameter range expressions. 
The iteration variables are bound to the indices of the received data, which is
interpreted as a multi-dimensional array in *row-major* order.

If the number of elements received is known, it is preferable to specify it explicitly in order
to allow for performance optimizations.

For example, the following code receives data from `stream_1` for `K` elements
and assigns the received data to the array `a`.
```
completion completion_name = foreach i32 k, f32 x in [0:K, receive(stream_1)] {
    a[k] = x;
}
```

The `completion_name` is a completion handle that may be used to wait for the completion of the `foreach` loop.
Note that the completion is triggered when the data has been received, not when it is sent.
After the completion triggers, the stream may be used for other sends or receives.

*Deadlocks*:
The sizes sent and received must match:

* Each `foreach` loops iterating over a stream that specifies the number of elements to receive,
must match the number of elements sent over the corresponding [`send`](#streaming-data-with-send) statement.

*Failure to correctly match the sizes sent and received may result in a deadlock.*


#### Processing arrays asynchronously with `map`

Inside a `compute` block, the `map` statement is used to apply a computation to each element of an array.
```rust
completion comp = map variables in [parameter_range_expression] {
  // Affine assignment statements
}
```
Every assignment statement must use an *affine expression* of the variables in the range expression.
The motivation for this is to ensure that the map can be efficiently vectorized.

There is no guarantee on the order in which the map is executed.
Hence, the map must not contain loop-carried dependencies.

For example,
```rust
completion comp = map i32 i, i32 j in [0:10, J] {
    a[i + 2 * j + 1] = i;
}
```

If you need to perform non-affine operations or exploit loop-carried dependencies, 
use a [`for`](#processing-arrays-sequentially-with-for) loop instead.

#### Await completions with `await`

Inside a `compute` block, an `await` statement is used to wait for a completion to trigger.
The `await` can be applied to a completion name.
```rust
await completion_name;
```
The `await` can be immediately applied to an asynchronous operation as a shorthand:
```
await operation;
// Is semantically equivalent to:
completion c = operation;
await c;
```

For example,
```rust
// Execute a map and wait for its completion
await map i32 i in [0:10] {
    // Statements
}
// Wait for completion of a send
await send(local_array, stream_name);
// Wait for completion of a receive
await foreach i32 k, f32 x in [0:K, receive(stream_name)] {
  // Statements
}
// Wait for a completion
await comp;
```

Note that statements inside an `await` may still be preempted by other asynchronous operations!
Awaiting the same completion twice is considered undefined behavior.

See the [Semantics of Asynchronous Statements](#semantics-of-asynchronous-statements) for more details.


#### Processing arrays sequentially with `for`

Inside a `compute` block, the `for` loop is used to apply a computation to each element of an array in a sequential order.

```rust
for variables in [parameter_range_expression] {
  // Statement
}
```
The range expression must be a parameter range expression.

The number of variables must match the number of dimensions in the parameter range expression.
The variables are bound to the indices given by the range expression.
The variables must be of type `i32`.

A `for` loop does not return completions, it executes sequentially and in-order.

For example, a `for` loop can exploit loop-carried dependencies:
```rust
for i32 i, i32 j in [0:I, 0:J] {
    a[i, j] = a[i-1, j] + a[i, j-1];
}
```
It can also use non-affine indexing:
```rust
for i32 i, i32 j in [0:I, 0:2] {
    a[i*j] = a[i] + a[j];
}
```

#### Computing asynchronously with `async`

Inside a `compute` block, an `async` block is used to execute a computation asynchronously.

```rust
completion comp = async {
  // Assignment statements or nested for-loops
}
```


### Phases

One may define multiple phases in a kernel.
Each phase may contain one or more `place`, `dataflow`, and `compute` blocks.

For a `place` block defined in the outermost scope, the fields defined therein are in-scope for all `compute` blocks in all phases.
For a `place` block within a phase, the fields defined therein are in-scope for the `compute` blocks in that phase.

Similarly, for a `dataflow` block defined in the outermost scope, the streams defined therein are in-scope for all `compute` blocks in all phases.
For a `dataflow` block within a phase, the streams defined therein are in-scope for the `compute` blocks in that phase.

Within each phase, there can be at most one `compute` block defined per PE.
If multiple `compute` blocks are defined per PE per phase, the behavior is undefined.
After each `compute` block, there is a set of implicit `await` statements
that wait for all completions to be triggered before starting the next `compute` block.
Note that this does *not* imply that all PEs have executed the `compute` block.
No `compute` block may be defined in the outermost scope.

Phases run in the order they are defined in the code from each PE's point of view.
That is, a PE goes through its phases in-order.
A PEs may participate in some phases and not in others.

For example:
```rust
place for i, j in [0:I, 0:J] {
    f32[K] a <- arg1[i, j, 0:K];
}

dataflow for i, j in [0:I, 0:J] {
  stream<f32> output = argument[i, j];
}

phase {
  place for i, j in [0:I, 0:J] {
    f32[K] b <- arg2[i, j, 0:K];
  }
   
  dataflow for i, j in [0:I, 0:J] {
    stream<f32> eastwards = relative_stream(1, 0);
  }
  
  compute for i, j in [0:I, 0:J] {
     // Within this compute block:
     // b and a are in scope, eastwards is in scope, and output is in scope
  }

}

phase {

  place for i, j in [1:I-1, 1:J-1] {
    f32[K] c <- arg3[i, j, 0:K];
  }

  dataflow for i, j in [1:I-1, 1:J-1] {
    // The communication pattern switches direction in this phase
    stream<f32> westwards = relative_stream(-1, 0);
  }
  
  compute for i, j in [1:I-1, 1:J-1] {
    // Within this compute block:
    // c is in scope, westwards, and output is in scope
  }

}
```

##  Semantics of Asynchronous Statements

Asynchronous statements may, but do not necessarily run in parallel.
Instead, they may execute in any order and may be interleaved with other statements.
An asynchronous statement may be pre-empted at any time, even partially during its execution.
Hence, it is imperative to properly define their semantics to avoid problems, such as *data races*
and properly define how the representation can be lowered to a task or thread model.

### The Happens-Before Graph

The asynchronous semantics can be defined in terms of a *happens-before* graph. 
For each statement `S` in a `compute` block, we define a *happens-before* relation `->` between statements.
Intuitively, `S1 -> S2` means that `S1` must complete before `S2` can start.

A blocking statement is a statement that must complete before
any following statement can start. In particular:
- `await` statements are blocking.
- assignments to fields are blocking.

We define the order in terms of the `await` statements in the code.
If `S1` is a non-blocking statement, let `c` be its completion.
We have that `S1 -> S2` if *any* of the following hold:
1. `S1` and `S2` are in the same compute block and one of the following hold:
   - (a) `S1` is a blocking statement and `S2` follows `S1` in all execution paths.
   - (b) `S1` is a non-blocking statement, and there is a statement `await c` between all possible execution paths from `S1` to `S2`.
2. `S1` is a `send` statement, and `S2` is the `await` statement of the corresponding `receive` forming the stream edge.
3. There exists a stream edge from some statement `S3` to `S4` and `S1 -> S3` and `S2` follows `S4` on all execution paths.
4. There is a statement `S3` where `S1 -> S3` and `S3 -> S2`.

Note that we handle phases by implicitly adding `await` statements for all outstanding 
completions at the end of each `compute` block.

Statements that are not ordered by happens-before are considered **concurrent**.

The happens-before graph is used to define data races and deadlocks.
Moreover, it can be used for lowering, specifically it can be used to 
determine how the code can be mapped to a task-based or thread-based model
and how to resolve `auto` routing declarations.

### Data Races

Writing to an array in a statement while concurrently reading from it 
or writing to it in another statement in the same 
compute block is considered a *data race*
and is considered undefined behavior. 
In particular, sending data from an array while concurrently
writing to it is considered a data race.
You must synchronize such statements using `await`.
The motivation for this strict definition is to ensure correctness regardless
of the interleaving of concurrent operations.

```rust
// For example, this is a data race:
// Concurrently writing to and sending from the same array
// We forbid this because the result of the send would be non-deterministic
completion c1 = send(a, stream);
for k in [0:K] {
    // Data Race!!
    a[k] = k; 
}
await c1;

// Correctly synchronized, we would get
completion c1 = send(a, stream);
await c1;
for k in [0:K] {
    a[k] = 1;
}

// Correctly synchronized with short-hand await syntax
await send(a, stream);
for k in [0:K] {
    a[k] = 1;
}

```

#### Example: Ping-Pong

Here is an example that synchronized through multiple compute
blocks using a ping-pong pattern:
It also includes one statement that demonstrates a data race.

```rust
phase {
  // Example: 'Ping-Pong'
  // Ping-pong pattern to synchronize two compute blocks
  // This is a correct way to synchronize two compute blocks
  // that write to the same array
  
  // Send from 1 to 0
  // Concurrently update array a
  // at 0, wait for receival, then send to 1
  // at 1, wait for receival
  // Then, update array a at 1
  
  place i, j in [0:2, 0] {
    f32[K] a;
  }

  dataflow i, j in [0:2, 0] {
      stream<f32> eastwards = relative_stream(1, 0);
      stream<f32> westwards = relative_stream(-1, 0);
  }

  compute i, j in [0, 0] {
     // S1
     completion c1 = foreach x, k in [receive(eastwards)] {
        a[k] = 2 * x
     }
     // S2
     await c1;
     // S3
     completion c2 = send(a, westwards);
  }

  compute i, j in [1, 0] {
     // S4
     completion c3 = send(a, eastwards);
     // S5 (data race)
     a[0] = 0;
     // S6
     completion c4 = foreach x, k in [receive(westwards)] {
        // S7 (correctly synchronized)
        a[k] = x;
     }
  }
}
```
Analysis of the Ping-Pong example above:
We have that `S4 -> S2` because of `await c1` statement `S2`.
We have `S2 -> S3` because an `await` blocks until it completes.  
We have that `S2 -> S7` because there is a stream edge from `S3` to `S6`
and all execution paths to `S7` go through `S6`.
Hence, we have `S4 -> S7` by transitivity.
Hence, the statement `S7` is correctly synchronized with `S4`.

However, the access at `S5` is concurrent with the `send` at `S4`.
This is a data race.


### Deadlocks

Some asynchronous statements cannot make progress until some event occurs.
It is guaranteed that if there exists a statement that
can make progress, at least one of them will make progress.
There is no guarantee of fairness, that is, concurrents statements may be
executed in any respective order and may be preempted at any time.
Failure to guarantee completion regardless of progress order of concurrent operations constitutes a **deadlock**.

For example, each iteration of a `foreach` waits until receiving a data element.
An `await` statement waits until a completion triggers. 
A `send` statement requires that the stream has space to carry the data
and may stall if the receiver is not ready to receive the data.
A deadlock-free program will ensure that all PEs can make progress
eventually regardless of the interleaving of statements.

## Semantics of Routing Declarations

Routing declarations must respect the limitations on how `channel`s are used.
Specifically, it must be avoided that two messages are routed through the same `channel` 
at the same PE simultaneously.

### The Routing Graph

The routing graph of a phase is a directed graph that describes how data is routed between PEs.
Note that the routing graph is defined in terms of the PE coordinates, so
its size grows with the size of the PE grid. It serves as a formal model for defining
the semantics, but should not be constructed explicitly.

Recall that stream edges are pairs of [send](#streaming-data-with-send) 
and [receive](#receiving-streaming-data-with-receive) operations that are matched across PEs.
Stream edges must not cross [phases](#phases), that is, a stream edge must be entirely contained within a phase.

The routing graph contains the following nodes *V*, edges *E*, and paths *P*:
- Each PE is a node in the graph.
- Consider each stream edge from PE `(x_1, y_1)` to PE `(x_2, x_2)` going through stream *C* on channel *L* through PE `hops = [(dx_1, dy_1), (dx_2, dy_2), ..., (dx_n, dy_n)]`.
We add an edge from `(x_1+dx_i, y_1+dy_i)` to `(x_1+dx_{i+1}, y_1+dy_{i+1})` for each *i* in *0, ..., n*.
where we use the convention that `dx_0 = dy_0 = 0`.
- Moreover, we add the resulting path `(x_1, y_1), ..., (x_1+dx_i, y_1+dy_i), ..., (x_2, y_2)` to the list of paths *P*
and record the stream *C* and channel *L*. 
Importantly, if two paths are identical and share the same stream, they are considered the same path and only one is recorded.
This can occur the same PE-pair uses the same stream multiple times in the same phase.
Such uses are already synchronized correctly using completions.

For example, the following code correctly sets up
a routing declaration for a 1D 2-phase reduce for 4 PEs:
It can use a single channel for both phases, as the streams
are properly sequenced in different phases.
```rust
// 1D 2-phase reduce for 4 PEs
place i, j in [0:4, 0] {
    f32[K] a;
}

phase {
  dataflow i32 i, i32 j in [0:4, 0] {
    stream<f32> hop1 = relative_stream(-1, 0) {
      hops = [(-1, 0)];
      channel = 0;
    };
  }
  compute i32 i, i32 j in [1:4:2] {
    send(a, hop1);
  }
  compute i32 i, i32 j in [0:4:2] {
    foreach i32 k, i32 x in [0:K, receive(hop1)] {
      a[k] += x
    }
  }
}

phase {
  dataflow i32 i, i32 j in [0:4, 0] {
    stream<f32> hop2 = relative_stream(-2, 0) {
      hops = [(-1, 0), (-1, 0)];
      channel = 0;
    };
  }

  compute i32 i, i32 j in [2, 0] {
    send(a, hop2);
  }

  compute i32 i, i32 j in [0, 0] {
    foreach i32 k, i32 x in [0:K, receive(hop2)] {
      a[k] += x
    }
  }

}
```

The routing graphs for this example contains 4 nodes, one for each PE.
In the routing graph for the first phase,
there are two edges from PE (1, 0) to PE (0, 0) and PE (3, 0) to PE (2, 0).
In the second phase,
There is a single edge from PE (2, 0) to PE (0, 0).

### Undefined Behavior

Next, we describe the condition under which the routing behavior is undefined:

We say that *P1* happens-before *P2* and write *P1 -> P2* if
the receive of *P1* happens-before the send of *P2*.

**If for a routing graph of a channel there are two paths *P1* and *P2* that share a PE `(x, y)`
and *P1* and P2 are not ordered by happens-before,
then the behavior is undefined.**

This is because the two messages may interfere with each other
and the order in which they are processed may become nondeterministic.
Recall that sending onto the same stream [must be synchronized using completions
to avoid data races](#streaming-data-with-send). Hence, sending through the same stream multiple times
in the same phases is ok as long as the sends (and receives) are correctly synchronized.

Keep in mind that PEs transition between phases asynchronously,
that is, a PE may advance to the next phase before another PE has completed the current phase.
We exploit here implicitly that routers back-pressure when
they receive data from a channel on which they are not configured
to receive. 
[A Note regarding potential extensions]
The current definition is tailored to the case
where all streams are point-to-point paths.
If multicasting is used, the correctness conditions become more challenging
to specify.

### Parametric Routing Graph

As the routing graph has a size that grows with the size of the PE grid,
we will not construct it explicitly.
Instead, we describe a *parametric* routing graph that describes the routing
graph in terms of predicated edges.

We again consider a particular phase.
The parametric routing graph of a phase is defined as follows:

There is a node for each compute block in the phase.
We rename all variables in the `compute` and `dataflow` subgrid expressions to `i` and `j` for simplicity.
The node is identified with the `i, j in subgrid_expression` that describes the PE coordinates of the compute block
and the two variables are bound to the PE coordinates in the compute block.
For example, `i, j in [0:I, 0:J]` could be a node in the routing graph.

The edges of the parametric routing graph are defined as follows:
Initialize a stack of vertex-index pairs to visit and a set of visited vertex-index pairs.
Add the node-index pair `(v, 1)` to the stack.
For each `send` statement with `hops = [(dx_1, dy_1), ..., (dx_h, dy_h)]`
in a compute block associated with the node `v`, we iteratively add edges as follows.

Until the stack is empty:
Pop the top vertex-index pair `(u, k)` from the stack.
Consider the current vertex `u=[I1:I2:I3, J1:J2:J3]` and the next hop `(dx_k, dy_k)` at index `k`.
Add an edge `(u, w)` to the following vertices `w` and 
if `(w, k+1)` is not in the visited set add `(w, k+1)` to the stack:

**Case: The stride is `I3 = J3 = 1`:**

- `[I1:I2:1, J1:J2:1]` with predicate (the cases are mutually exclusive by definition because `|dx_k|+|dy_k|==1`):
  - `i + 1 < I1 - 1` if `dx_k = 1`
  - `i - 1 > I1` if `dx_k = -1`
  - `j + 1 < J1 - 1` if `dy_k = 1`
  - `j - 1 > J1` if `dy_k = -1`

- If `dx_k != 0`, to all blocks `[I4:I5:1, J4:J5:1]` for which `J4 < J2`, `J5 >= J1`, and
  - for which `I4 = I2 + 1` with the predicate `i = I2 && J4 <= j < J6` if `dx_k == 1`
  - for which `I5 = I1 - 1` with the predicate `i = I1 && J4 <= j < J6` if `dx_k == -1`
  - Note that the ranges `J4:J5` of all such blocks must together cover the range `J1:J2`.
Failure to do so constitutes an incorrect declaration of stream edges (deadlock).

- Proceed symmetrically for `dy_k != 0`.


**Case: The stride is > 1, but the compute blocks connected by a stream edge
have the same strides as each other:**

- If `dx_k != 0`, to all blocks `[I4:I5:I3, J4:J5:J3]` for which `J4 < J2`, `J5 >= J1`, and 
for which `I4 = I2 + dx_k (mod I3)` with the predicate `I4 <= i + 1 < I5 && J4 <= j < J6`.
  - Note that the ranges `J4:J5` of all such blocks must together cover the range `J1:J2`, and similarly, 
the ranges `I4:I5` must together cover the range `I1+dx_k:I2+dx_k`.
Failure to do so constitutes an incorrect declaration of stream edges (deadlock).

- Proceed symmetrically for the case `dy_k != 0`.

**Case: The strides are either 1 or the same value.**

TODO (Needed? maybe convenient for larger boundary conditions)

#### Analysis

Note that blocks with a single element can be interpreted as having any arbitrary stride.
This is useful for implementing boundary conditions.

Runtime: Note that each vertex is added to the stack at most `h` times, where `h` is the number of hops.
Adding all edges for a given vertex takes at most `n` time,
where `n` is the number of vertices in the parametric routing graph.
Hence, the overall runtime is `O(n^2 * h)`.

The construction of the parametric routing graph also validates the correctness of stream edges.
If at any point the target vertex does not exist, the program is incorrect due to a deadlock,
which is raised as an error by the compiler.


### The Conflict Graph

The conflict graph can be used to determine if a routing declaration is correct,
and resolve the `auto` routing declarations.
The conflict graph is a directed graph that describes the conflicts between streams.
Two streams conflict if they are routed through the same channel at the same PE.

We use the parametric routing graph to construct the conflict graph.



## Examples

### Vertical Advection


```rust
kernel vadv<I,J,K>(stream<f32>[I, J] utens_stage,
                  stream<f32>[I, J] readonly u_stage,
                  stream<f32>[I, J] readonly wcon,
                  stream<f32>[I, J] readonly u_pos,
                  stream<f32>[I, J] readonly utens,
                  stream<f32>[I, J] writeonly datacol,       
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

    // Set up communication streams
    // The only PE-PE communication is for wcon, which is sent to the west  
    // We additionally set up a stream for the inputs and outputs
    dataflow i, j in [0:I, 0:J] {
      stream<f32> u_stage_c = u_stage[i, j];
      stream<f32> wcon_c = wcon[i, j];
      stream<f32> u_pos_c = u_pos[i, j];
      stream<f32> utens_c = utens[i, j];
      stream<f32> utens_stage_c = utens_stage[i, j];
    }
    
    // Copy the data
    
    compute i, j in [0:I, 0:J] {
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
      # Local fields live inside the phases scope
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
      f32 divided;
    }
  
    // Set up communication streams
    // The only PE-PE communication is for wcon, which is sent to the west  
    // We additionally set up a stream for the outputs
    dataflow i, j in [0:I, 0:J] {
      stream<f32> westwards = relative_stream(-1, 0);
      
      stream<f32> utens_stage_c = utens_stage[i, j];
      stream<f32> datacol_c = datacol[i, j];
    }
  
    ////
    // Computation
  
    compute i, j in [0, 0:J] {
      // Boundary condition
      // ...
      foreach k, x in [0:K, receive(westwards)] {
        // ...
      }
    }
  
    compute i, j in [1:I, 0:J] {
        
        send(wcon_local[0:1], westwards);
        send(wcon_local[1:K], westwards);

        // base of the forward
        await foreach i32 k, f32 x in [0:1, receive(westwards)] {
            gav[k] = -0.25 * x * wcon_l[k];
            gcv[k] = 0
            // ...
        }

        // Forward pass: data movement
        await foreach i32 k, f32 x in [1:K, receive(westwards)] {
          gav[k] = -0.25 * x * wcon_l[k];
          gcv[k-1] = 0.25 * x * wcon_l[k];

          as_[k] = gav[k] * bet_m;
          cs[k] = gcv[k] * bet_m;
          acol[k] = gav[k] * bet_p;
          ccol[k] = gcv[k] * bet_p;
          bcol[k] = dtr_stage - acol[k] - ccol[k];

          correction_term[k] = -as_[k] * (u_stage_l[k-1] - u_stage_l[k]) - cs[k] * (u_stage_l[k+1] - u_stage_l[k]);
          dcol[k] = dtr_stage * u_pos_l[k] + utens_l[k] + utens_stage_l[k] + correction_term[k];

          // Thomas forward
          divided = 1.0 / (bcol[k] - ccol[k-1] * acol[k]);
          ccol_2[k] = ccol[k] * divided;
          dcol_2[k] = (dcol[k] - dcol[k-1] * acol[k]) * divided;
        }
  
        // Boundary condition (k=K-1) not shown
        for i32 k in [K-1] {
          /// Boundary condition ...
        }
        // Main backwards loop          
        for i32 k in [K-2:0:-1] {
          datacol_l[k] = dcol_2[k] - ccol_2[k] * datacol_l[k+1];
          utens_stage_l[k] = dtr_stage * (datacol_l[k] - u_pos_l[k]);
        }
  
        // Copy the data to the output
        send(datacol_l, datacol_c);
        send(utens_stage_l, utens_stage_c);
    }
  }

}


```


### 2D Laplacian

```rust
kernel laplacian<I,J,K> (stream<f32>[I+2, J+2] readonly in_field,
                         stream<f32>[I, J] writeonly lap_field) {
    
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
      stream<f32> in_field_l = in_field[i, j];
    }
  
    compute i, j in [0:I+2, 0:J+2] {
      foreach k, x in [0:K, receive(in_field_l)] {
        local_input[k] = x;
      }
    }
  }
  

  phase {
    // Set up communication streams
    dataflow i, j in [0:I+2, 0:J+2] {
       stream<f32> eastwards = relative_stream(+1, 0);
       stream<f32> westwards = relative_stream(-1, 0);
       stream<f32> northwards = relative_stream(0, -1);
       stream<f32> southwards = relative_stream(0, +1);
       
    }

    dataflow i, i in [1:I+1, 1:J+1] {
        stream<f32> lap_field_l = lap_field[i, j];
    }
  
    // Edge senders
    compute i, j in [0, 1:J] {
        // Streaming send to the right
        send(local_input, eastwards);
        // We receive nothing
    }
    
    compute i, j in [I+1, 1:J] {
        // Streaming send to the left
        send(tosend, westwards);
        // We receive nothing
    }
    // ...
  
    compute i, j in [1:I+1, 1:J+1] {
        // Streaming parallel computation (map)
        completion f = map i32 k in [0:K] {
            local_result[k] = local_input[k] * 4;
        }
  
        // No data race, both map and send are reading from local_input
        send(local_input, westwards);

        // Example of an await for a completion.
        await f;
        
        // Writing to local_result form the map would be considered a data race.
        await foreach i32 k, f32 x in [0:K, receive(westwards)] {
          local_result[k] -= x;
        }

        // Writing to the same array from multiple foreach blocks concurrently
        // is considered a data race.
        // Hence, we need to run one after the other.
        send(local_input, eastwards);
        await foreach i32 k, f32 x in [0:K, receive(eastwards)] {
          local_result[k] -= x;
        }
        // ...

        // after all computation has finished:
        await send(local_result, lap_field_l);   
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
kernel conv<J>(stream<f32>[J] readonly input,
               stream<f32>[J] writeonly output,
               f32[3] readonly KERNEL) {

    // Data placement
    place i, 0 in [0:J, 0] {
        f32 y;
        f32[3] kernel <- KERNEL;
    }

    // Communication
    dataflow i, j in [0:J, 0] {
        stream<f32> eastwards = relative_stream(1, 0);
        stream<f32> westwards = relative_stream(-1, 0);
        stream<f32> input_local <- input[i];
        stream<f32> output_local -> output[i];
    }

    // Computation
    compute i, j in [1:J-1, 0] {

        // Streaming receive
        // Each PE receives a single scalar per time step
        foreach x in receive(input_local) {

            y = x * kernel[1];

            // Send the data to the right
            comp_east = send(x, eastwards);

            await foreach k, x2 in [0:1, receive(eastwards)] {
              y = y + x2 * kernel[0];
            }

            await comp_east;

            // Send the data to the left
            comp_west = send(x, westwards);

            await foreach k, x3 in [0:1, receive(westwards)] {
               y = y + x3 * kernel[2];
            }

            await comp_west;

            // Send the result to the output
            await send(y, output_local);
        }
    }

    // Left corner
    compute i, 0 in [0, 0] {
        // Streaming receive
        foreach x in receive(input_local) {
            // Send the data to the right
            comp_east = send(x, eastwards);

            y = x * kernel[1];

            await foreach x, y in [0:1, receive(westwards)] {
                y = y + x * kernel[2];
            }

            await comp_east;
            
            // Send the result to the output
            send(y, output_local);
        }
    }

    // Right corner
    // ...

}
```