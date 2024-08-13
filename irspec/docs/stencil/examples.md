
Consider the following laplacian stencil program expressed in gtscript:

```python
@gtscript.stencil(backend=cartesian_backend)
def lap_cartesian(
    inp: gtscript.Field[dtype],
    out: gtscript.Field[dtype],
    ext: gtscript.Field[dtype]
):
    with computation(PARALLEL), interval(0, None):
        out = -4.0 * inp[0, 0, 0] + inp[-1, 0, 0] + inp[1, 0, 0] + inp[0, -1, 0] + inp[0, 1, 0]
        ex_t = out[0, 0, 0] + out[0, 1, 0]
```

A direct translation results in the following computation:

```mlir
// Version with ? extents.
%res = spst.computation (%in) 
{
    schedule: PARALLEL,
    interval: [0:None, 0:None, 0:None]
} :
field<[?,?,?], {(?, ?, ?)}, f32> -> field<[?,?,?], {(?, ?, ?)}>, f32> {
    %out_1 = spst.statement(%in) : field<[?,?,?], {(?, ?, ?), f32} -> field<[?,?,?], {(?, ?, ?), f32}> {
            return -4.0 * %in[0, 0, 0] + %in[-1, 0, 0] + %in[1, 0, 0] + %in[0, -1, 0] + %in[0, 1, 0]
        }
    %out_2 = spst.statement(%out_1) : field<[?,?,?], {(?, ?, ?), f32}> -> field<[?,?,?], {(?, ?, ?), f32}> {
            return %out_1[0, 0, 0] + %out_1[0, 1, 0]
        }
    return %out_2
}
```

Performing type inference on the extents results in the following:

```mlir
// Version with inferred extents.
%res = spst.computation (%in) 
{
    schedule: PARALLEL,
    interval: [0:None, 0:None, 0:None]
} : field<[?,?,?],
    {(0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)
                (0, 1, 0), (-1, 1, 0), (1, 1, 0), (0, 2, 0)}, f32>
  -> field<[?,?,?], {(?, ?, ?)}, f32>
{
    %out_1 = spst.statement(%in)
            : field<[?,?,?], {(0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)
                                                            (0, 1, 0), (-1, 1, 0), (1, 1, 0), (0, 2, 0)}, f32>
            -> field<[?,?,?], {(0, 0, 0), (0, 1, 0)}, f32>
        {
            return -4.0 * %in[0, 0, 0] + %in[-1, 0, 0] + %in[1, 0, 0] + %in[0, -1, 0] + %in[0, 1, 0]
        }
    %out_2 = spst.statement(%out_1) 
            : field<[?,?,?], {(0, 0, 0), (0, 1, 0)}, f32>
            -> field<[?, ?, ?], {(0, 0, 0)}, f32> 
        {
            return %out_1[0, 0, 0] + %out_1[0, 1, 0]
        }
    return %out_2
}
```
Note the use of shorthand notation for the field types.


If we insert a `materialize` in between the two statements,
the extents no longer propagate across the boundaries.
This indicates a schedule where no recomputation is done.
Instead, the values are explicitly communicated.

```mlir
// Version with materialize to prevent recomputation
%res = spst.computation (%in) 
{
    schedule: PARALLEL,
    interval: [0:None, 0:None, 0: None]
} : field<[?,?,?],
          {(0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)
           (0, 1, 0), (-1, 1, 0), (1, 1, 0), (0, 2, 0)}, f32>
  -> field<[?,?,?], {(?, ?, ?)}, f32>
{
    %out_1 = spst.statement(%in) 
            : field<[?,?,?],
              {(0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)}, f32>
            -> field<[?,?,?], {(0, 0, 0)}, f32>
        {
            return -4.0 * %in[0, 0, 0] + %in[-1, 0, 0] + %in[1, 0, 0] + %in[0, -1, 0] + %in[0, 1, 0]
        }
    %out_mat = spst.materialize(%out1) : field<[?,?,?], {(0, 0, 0)}, f32>
                                         -> field<[?,?,?], {(0, 1, 0)}, f32>
    %out_2 = spst.statement(%out_1, %out_mat)
        : field<[?,?,?], {(0, 0, 0)}, f32>,
          field<[?,?,?], {(0, 1, 0)}>, f32> 
        -> field<[?,?,?], {(0, 0, 0}>, f32> {
            return %out_1[0, 0, 0] + %out_mat[0, 1, 0]
        }
    return %out_2
}
```

The materialize operation is used to prevent recomputation of the value.