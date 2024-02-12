# Spatial Stencil Compiler



## Stencil IR

The goal of this document is to give an overview of the key concepts present in the IR.
It does not (yet) fully describe the semantics of the computation.

### Key Concepts

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
i<8>, i<16>, i<32>,
```
the unsigned integer types
```
u<8>, u<16>, u<32>,
```
and the floating point types
```
f<16>, f<32>
```

Note that not all targets might support all scalar types natively.

### Domain types

For every triple of positive integers `x, y, z`, there is a domain type

```
stencil.domain<x, y, z>
```

If one or more dimensions of the domain are unknown,
they may be replaced with a `?` placeholder:
```
stencil.domain<?, ?, ?>
stencil.domain<x, ?, ?>
stencil.domain<x, y, ?>
...
```
The purpose of placeholders is to allow type-inference to deduce the
domain sizes. To lower the representation, we require all dimensions
of the iteration domain have been inferred at compile time.

### Extent types

An extent defines the access offsets of a stencil operation.
For every integer triple `i, j, k` there is an extent access:
```
stencil.access<i, j, k>
```
If an extent is unknown, it can contain the placeholder extent `?`. The purpose of placeholders is to allow
type-inference to deduce the extent accesses. We require the extent accesses to be compile-time inferrable.

A sequence of extent accesses forms the extent:
```
stencil.extent<extent-access* >
```

### Field Types

For every domain type D, scalar type T, and extent type E,
there is a field type
```
stencil.field<D, E, T>
```
that models a multi-dimensional array (field) over iteration domain D with extent E and scalar type T


### Interval Types

A  type defines inclusive lower intervals and exclusive upper intervals for an iteration. Both upper and lower intervals are optional.

For every pair of integers `a` , `b` we have types:
```
stencil.interval<a, b>
stencil.interval<a, None>
stencil.interval<None, b>
stencil.interval<None, None>
```
We may also use placeholder values `?` to indicate a fixed bound that has yet to be inferred:
```
stencil.interval<?, ?>
stencil.interval<?, None>
stencil.interval<None, ?>
stencil.interval<None, None>
```
The purpose of this type system is to allow representation of both implicit padding and explicit padding.
After type-inference all placeholders must have been removed.

### Schedule Type

There are three possible schedules

```
stencil.schedule-type<PARALLEL | FORWARD | BACKWARD>
```

## Operations

We use the following notation, borrowed from MLIR:
```
result = operation(inputs) { properties } : input-types -> output-types { contained-region }
```
Properties are a kind of compile-time known attribute associated with an operation.

### stencil.statement

```
%out = stencil.statement(%in_1, ... %in_n) {
    } : field [D, E1, T1] , ...,  field [D, E_n, T_n] -> field [D, E_0, T_0] {
        statement
    }
```

The statement may only contain references to `in_1, ... in _n` at the specified extents.

A field that is the result of a stencil statement is an intermediate field.
For any field, only accesses that are within their extent type are permitted within the statements.
Statements accessing a field with placeholders in their extent type always type check.

Note that the output type's extent `E_0` defines which values are available for subsequent statements.
In particular, if every output value is computed only once, then the extent is `extent[(0, 0, 0)]`.
If for example every output computes also the right-neighbor, then the extent is `extent[(0, 0, 0), (0, 1, 0)]`.

### stencil.computation

A stencil accesses a set of input fields `in_1, ... in_n` and produces a set of output fields `out_1, ...., out_n`.
It has a property `schedule` which defines the order in which the last dimension is traversed.
The interval defines the iteration range with a value of type `interval`.
for each dimension `x`, `y`, `z`.

_**Note**: This representation allows us to both handle implicitly padded domains
(through the use of `?` domains and intervals), as well as explicitly padded domains._

```
%out_1, ..., %out_m = stencil.computation(%in_1, ..., %in_n) {
           schedule = s : schedule-type,
           interval = [x: interval, y: interval, z: interval]
   } : field [D, E_1, T_1] , ...,  field [D, E_n, T_n]
 -> field [D, E_n+1, T_N+1] , ...,  field [D, E_n+m, T_n+m] {
   stencil-statement-block
}
```

A stencil-statement-block is a block that contains one or more stencil.statements.
The execution semantics is equivalent to executing one statement after the other.

Note that the extent types, interval types, and domain type `D` must be compatible.
That is, any extent access plus an interval must not go beyond the bounds of the domain type `D`.

Moreover, the extent types of the intermediate fields must be compatible with the access types of their
consuming statements.

### stencil.conditional

[TODO Not (yet) supported]

### stencil.while

[TODO Not (yet) supported]

### stencil.program

A stencil program contains one ore more stencil computations.

```
%out_1, ..., %out_m = stencil.program(%in_1, ..., %in_n) {
   } : field [D, E_1, T_1] , ...,  field [D, E_n, T_n]
 -> field [D, E_n+1, T_N+1] , ...,  field [D, E_n+m, T_n+m] {
   stencil-computation-block
}
```

The outputs of individual stencil computations define intermediate fields, which may
serve as inputs to further computation blocks.

_TODO: Add spatial mapping attributes or field types. I would probably define the spatial mapping as additional attributes to the operations (stencil.program in particular). 
The alternative is to create a "mapped field" type that additionally contains information about how it is mapped onto the device.
This would allow putting different fields onto different locations, so it's more general._