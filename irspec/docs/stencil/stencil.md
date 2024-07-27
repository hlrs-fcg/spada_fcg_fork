# Stencil IR

The goal of this document is to give an overview of the key concepts present in the IR.
It does not (yet) fully describe the semantics of the computation.

This document (loosely) follows the MLIR conventions of describing IR as a combination of
types, operations, attributes, basic blocks, and regions. By default, all blocks are in SSA CFG form.

### Types

We use the following notation for types
```
type-name<type-arguments>
```
The type arguments can either be literals or other types.

#### Scalar Types

The scalar types include

the signed integer types
```
i8, i16, i32,
```
the unsigned integer types
```
u8, u16, u32,
```
and the floating point types
```
f16, f32, f64,
```
and boolean types
```
bool
```

Note that not all targets might support all scalar types natively.

#### Domain types

The abstract base domain type is
```
spst.domain
```

For every triple of positive integers `x, y, z`, there is a domain sub-type

```
spst.cartesian<x, y, z>
```

If one or more dimensions of the domain are unknown,
they may be replaced with a `?` placeholder:
```
spst.cartesian<?, ?, ?>
spst.cartesian<x, ?, ?>
spst.cartesian<x, y, ?>
...
```
The purpose of placeholders is to allow type-inference to deduce the
domain sizes. To lower the representation, we require all dimensions
of the iteration domain have been inferred at compile time.

#### Extent types

We utilize the type system to represent the access offsets of a stencil operation.
The goal is that the type encapsulates the data layout and data movement.

An extent defines the access offsets of a stencil operation.
For every integer triple `i, j, k` there is an extent access:
```
spst.access<i, j, k> ::= (i, j, k)
```
If an extent is unknown, it can contain the placeholder extent `?`. The purpose of placeholders is to allow
type-inference to deduce the extent accesses. We require the extent accesses to be compile-time inferrable.
Replacing a concrete value with a `?` creates a super-type. That is, the type `spst.access<?, ?, ? >` is a
super-type of all stencil accesses.

A sequence of extent accesses forms the extent:
```
spst.extent<extent-access* >
```
This sequence must *contain no duplicates*.

An extent `E_sub` where every access is a subtype of an access of another extent `E_sup` is a sub-type of `E_sup`.
As such, values of type `E_sub` may be consumed everywhere that values of type `E_sup` may be consumed.

Example:
```
%x_sup : spst.extent<(0, 0, 0)>
%x_sub : spst.extent<(0, 0, 0), (0, 0, 1)>
// x_sub is a sub-type of x_sup
%y : spst.extent<(0, 0, ?)>
// y is a super-type of both x_sup and x_sub
```

For example, in the following program, the extent of `out` is `spst.extent<(0, 0, 0)>`:
and valid extents of inp are:

```
inp : extent<(?, ?, ?)>
inp : extent<(?, ?, 0)>
inp : extent< (-1, 0, 0), (0, 0, 0), (0, -1, 0), (0, 1, 0), (1, 0, 0)>
```


#### Field Types

For every domain type D, scalar type T, and extent type E,
there is a field type
```
spst.field<D, E, T>
```
that models a multi-dimensional array (field) over iteration domain D with extent E and scalar type T.

A field type `spst.field<D, E1, T>` is a subtype of a type `spst.field<D, E2, T>` if `E1` is a subtype of `E2`.
The domains and scalar types must match.

#### Interval Types

An interval type defines an inclusive lower bound and exclusive upper bound for an iteration. Both upper and lower bound are optional.

For every pair of integers `a` , `b` we have types:
```
spst.interval<a, b>
spst.interval<a, None>
spst.interval<None, b>
spst.interval<None, None>
```
We may also use placeholder values `?` to indicate a fixed bound that has yet to be inferred:
```
spst.interval<?, ?>
spst.interval<?, None>
spst.interval<None, ?>
spst.interval<None, None>
```
As before, a placeholder value creates a super-type.
In particular, all intervals are subtypes of `spst.interval<?, ?>`.

The purpose of this type system is to allow representation of both implicit padding and explicit padding.
After type-inference all placeholders must have been removed.

#### Schedule Type

There are three possible schedules

```
spst.schedule<PARALLEL | FORWARD | BACKWARD>
```

## Operations

We use the following notation, borrowed from MLIR:
```
result = operation(inputs) { properties } : input-types -> output-types { contained-region }
```
Properties are a kind of compile-time known attribute associated with an operation.

If an operation has no properties, it may use the shorthand
```
result = operation(inputs) : input-types -> output-types { contained-region }
```

### spst.return

The `spst.return` operation is used to return a value to the containing operation.
The semantics depends on the containing operation.

```
spst.return %value : T
```


### spst.statement

```
%out = spst.statement(%in_1, ... %in_n) :
       spst.field<D, E1, T1> , ...,   spst.field<D, E_n, T_n> ->  spst.field<D, E_0, T_0> 
{
    spst.return expression : T
}
```
A field that is the result of a stencil statement is an intermediate field.


```
field_expression ::= field[constant_literal, constant_literal, constant_literal]
expression ::= constant_literal | parameter_literal | field_expression | expression + expression | expression - expression | expression * expression | expression / expression | expression // expression | expression % expression | (expression) 
bool_expression ::= expression == expression | expression != expression | expression < expression | expression <= expression | expression > expression | expression >= expression | bool_expression & bool_expression | bool_expression | bool_expression | !bool_expression | (bool_expression)
```
where `//` denotes integer division and `%` denotes modulo,
and `==`, `!=`, `<`, `<=`, `>`, `>=`, `&`, `|`, `!` are the standard comparison and logical operators.

The output type's extent `E_0` defines which values are available for subsequent statements.
In particular, if every output value is computed only once at `(0, 0, 0)`, then the extent is ` spst.extent<(0, 0, 0)>`.
If also the right-neighbor is available, then the extent is ` spst.extent<(0, 0, 0), (0, 1, 0)>`.

The expression may only contain accesses to the fields `in_1, ... in _n` at the allowed extents.
For any field, only accesses that are within their extent type are permitted within the statements.
Which accesses are permitted is relative to the output's extent type.
Specifically, every access to `(i, j, k)` is added to every output extent `(x, y, z)`
and results in an access to `(i+x, j+y, k+z)`.

This corresponds to a Minkovski sum of the accesses of the statements and the accesses of the output's extent type._
The arguments to the statement must be a sub-type of the union of all these accesses.
Specifically, accessing a field with placeholders in their extent type always type checks.

### spst.computation

A stencil accesses a set of input fields `in_1, ... in_n` and produces a set of output fields `out_1, ...., out_n`.
It has a property `schedule` which defines the order in which the last dimension is traversed.
The interval defines the iteration range with a value of type `interval`.
for each dimension `x`, `y`, `z`.

_**Note**: This representation allows us to both handle implicitly padded domains
(through the use of `?` domains and intervals), as well as explicitly padded domains._

```
%out_1, ..., %out_m = spst.computation(%in_1, ..., %in_n) {
           schedule = s : spst.schedule,
           interval = [x: spst.interval, y: spst.interval, z: spst.interval]
   } : spst.field<D, E_1, T_1> , ...,  spst.field<D, E_n, T_n>
 -> spst.field<D, E_n+1, T_N+1> , ...,  spst.field<D, E_n+m, T_n+m> {
   stencil-statement-block
}
```

A stencil-statement-block is a block that contains one or more `spst.statement`s and `spst.if`s.
The execution semantics is equivalent to executing one statement after the other.

Note that the extent types, interval types, and domain type `D` must be compatible.
That is, any extent access plus an interval must not go beyond the bounds of the domain type `D`.

The input argument's extent types must be super-types of the consuming statements.
Similarly for the extent types of intermediate fields.

Every stencil-statement block must be terminated with a spst.return operation.

### spst.materialize

By default, intermediate fields are not materialized, instead
their values are computed from the input fields at every offset
that is consumed. To reduce computation costs, one might instead
want to materialize an intermediate and send its value to the
consuming grid points.

The `spst.materialize` operation implements this.
Structurally, this manifests as a type cast from the input field
to the output field.
Without materialization, the extents propagate in a way that every intermediate
field is re-computed for each of its (distinct) consuming accesses. 
A `spst.materialize` may interrupt this propagation of extent type by 
means of a type-cast.
    
Example:
```
%mat = spst.materialize(%intermediate) : spst.field<D, spst.extent<(0, 0, 0)>, T> -> spst.field<D, spst.extent<(1, 1, 0)>, T>
```
In the example, consuming statement of `%mat` may access `%mat[1, 1, 0]` without affecting the type of `%intermediate`.

Semantically, this has the effect that whenever `%mat` is consumed,
it corresponds to reading information about `%intermediate` from other grid
points at the appropriately translated offsets.
In other words, the intermediate field `%intermediate` becomes materialized
and accessible via the `mat` variable.

Note that it is valid to use placeholder extents `?` in the types
of a materialize, as the actual values can be inferred based on the
producing and consuming statements and fields. Specifically, 
a materialize does not introduce any constraints on the input type.
The output type is deduced as a sub-type that can be substituted
into the consuming statements.

Example:
```
%mat = spst.materialize(%intermediate) : spst.field<D, spst.extent<(?, ?, ?)>, T> -> spst.field<D, spst.extent<(?, ?, ?)>, T>
...
%mat[0, 1, 0]
...
%mat[1, 0, 0]
...

// Type inference leads to 

%mat = spst.materialize(%intermediate) : spst.field<D, spst.extent<(?, ?, ?)>, T>
        -> spst.field<D, spst.extent<(0, 1, 0), (1, 0, 0)>, T>


```

### spst.if

The spst.if operation represents an if-then-else construct for conditionally executing one or more regions of code.
The operand to an if operation is a boolean field or integer field. For example:
```
spst.if (%mask) : spst.field<D, spst.extent<(0,0,0)>, bool> {
  ...
} else {
  ...
}
```
The conditional may include any number of `elif` branches
between the if and the else branch:
```
spst.if (%mask1) {
  ...
} elif (%mask2) {
  ...
} else {
  ...
}
```

spst.if may also produce one or more results. For example, when the if-else is used to represent
a ternary expression. 
Which values are returned depends on which execution path is taken at each grid
point of the domain, as indicated by the masks.

The “then” region has exactly 1 block. The “else” region may have 0 or 1 block.
In case the spst.if produces results, the “else” region must also have exactly 1 block.
The blocks are always terminated with spst.return.

Example:
```
spst.if (%mask)  {
  ...
}
```
The types of the yielded values must match the result types of the spst.if.


??? example "Example: If-else"
    ```
    %x = spst.if (%mask): spst.field<D, extent<(0,0,0)>, bool> -> spst.field<D, E_1, T_1> {
      %x_1 = ...
      spst.return %x_1 : spst.field<D, E_1, T_1>
    } else {
      %x_2 = ...
      spst.return %x_2 : spst.field<D, E_1, T_1>
    }
    ```

???+ example "Example: Ternary Choice"
    For example, to implement a ternary choice on the value of a field, the following pattern is used:
    
    ```
    %mask1 = spst.statement(%u) : spst.field<D, E1, T1> -> spst.field<D, E1, bool> { spst.return %u[0, 0, 0] > 0 }
    %mask2 = spst.statement(%u) : spst.field<D, E1, T1> -> spst.field<D, E1, bool>  { spst.return %u[0, 0, 0] < 0 }
    
    %x = spst.if (%mask1) {
      ...
    } elif (%mask2) {
      ...
    } else {
      ...
    }
    ```



!!! danger "No Side Effects"
    The spst.if operation and any nested operations are not allowed to have side effects. 
    The operation must not modify any state or memory that is accessed outside of the operation itself.
    This is necessary to ensure that the operation can be executed in parallel.

### spst.while

[TODO Not (yet) supported]

### spst.program

A stencil program contains one more stencil computations.
Its arguments and return types may contain both scalar types and field types.

```
%out_1, ..., %out_m = spst.program @programname (%in_1, ..., %in_n) {
   } : T_1, ...,  T_n
 -> T_N+1 , ...,  T_n+m {
   stencil-computation-block
}
```

The outputs of individual stencil computations define intermediate fields, which may
serve as inputs to further computation blocks.

The field input argument's extent types must be super-types of the consuming computations.
Similarly for the extent types of intermediate fields.
