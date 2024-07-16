# Parametric Semantic Representations

So far, we have introduced and discussed the formal definitions of [stream edges](#stream-edges),
the [happens-before graph](#the-happens-before-graph), and the [routing graph](#the-routing-graph).
Next, we discuss how to construct compact, *parametric* representations of these graphs.
The advantage of these representations is that there size is much smaller
than the size of the program grid, and constructing them is polynomial time in 
the size of the program.

## Constructing the Local Order

The program can be extracted from the control [flow graph](https://en.wikipedia.org/wiki/Control-flow_graph)
of basic blocks.
Then, compute the [Dominators](https://en.wikipedia.org/wiki/Dominator_(graph_theory))

If `S1` and `S2` are in the same basic block, then `S1 --> S2` if

- `S1` is blocking and `S2` follows `S1` in the basic block.
- `S1` is non-blocking and there is an `await` on the completion of `S1` between `S1` and `S2`.

Otherwise, `S1 --> S2` if:

- `S1` is blocking and `S1` dominates `S2`.
- The set of dominators of `S2` contains an `await` on the completion of `S1`.

Efficient and practical algorithms [can compute dominators in near-linear time](https://www.researchgate.net/publication/220639563_Finding_Dominators_in_Practice).


## *Parametric* Stream Edges

The parametric stream edges are a compact representation of the stream edges.
Each stream edge is represented as `(S1, (i, j)) , (S2, (i+dx, j+dy))` for statements
`S1`, `S2`, predicated by `P1`.
Here, `i` and `j` are variables that appear in the predicate.
The interpretation is that if the predicate `P1` is true for some `i`, `j` where
`(i, j)` is in the compute block of `S1`, then the stream edge exists for the PE `(i, j)`.

The first step to constructing stream edges is to determine the
order of the `send`s and `receive`s that occur to the same stream within each
`compute` block. This follows immediately from the local order `-->`.

*The following assumes that `compute` blocks and `dataflow` blocks
match N-1, that is, each compute block is specified by a single dataflow block
from its phase and a single global dataflow block.
We can remove this assumption by splitting the compute block into multiple blocks
until the condition is satisfied. (TODO: How? - OR: is there a direct way?)*

Next, we consider each `compute` block in a phase.
We rename all variables in the `compute` and `dataflow` subgrid expressions to use `i` and `j` for simplicity.
A `compute` block is now identified with some set of PEs defined as `i, j in [I1:I2:I3, J1:J2:J3]`.
Within this block, consider some `send` statement `S1`, which is the k-th `send` statement to its stream in the local order.
Let `(dx, dy)` be the offset of its stream. That is, a PE `(i, j)` sends to PE `(i+dx, j+dy)`.
Note that we assume the strides are positive without loss of generality.

We now construct the stream edges for this `send` statement.
For this, we observe the following constraints on the blocks `[I4:I5:I6, J4:J5:J6]` that can receive from the stream
when sending from PE `(i, j)`:

Range constraints:
- `I4 <= i + dx < I5`
- `J4 <= j + dy < J5`

Congruence constraints:
- `i + dx = I4 (mod I6)` in case `I6 > 1`
- `j + dy = J4 (mod J6)` in case `J6 > 1`

To correctly identify the stream edges, we need to check if there exists an `(i, j)` in the current
`compute` block for which all constraints are satisfied.

For this, first solve the linear congruence relations for `i` and `j` (if `I6 > 1` and `J6 > 1` respectively).
We use that `i=I1+x*I3` and `j=J1+y*J3` for some `x` and `y`.
Then, we need to solve for `x` and `y` in the following equations:
- `x * I3 = (I4 - I1 - dx) mod I6`
- `y * J3 = (J4 - J1 - dy) mod J6`


Let's focus on the first equation, as the second is symmetric.

1. Compute `gcd(I_3, I_6) = d` using the [Euclidean Algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm).
2. Check if `d` divides `I_4 - I_1 - dx`. If not, no solution exists (there is no stream edge).
3. If `d` divides `I_4 - I_1 - dx`, we can solve the equation:
   - Simplify the equation by dividing everything by `d`.
   - Solve the simplified equation using the
   [Extended Euclidean Algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm) to find one solution `x_0`:
   - The general solution is:
     `x = x_0 + k (I_6/d) for k = 0, 1, ... , d-1`

We filter out all solutions for which `I1 + x * I3 >= I2` as they are out of bounds of the `compute` block.

Now, we can apply the range constraints using the general solution of `x`
to determine the solutions `I1 + x * I3 + dx` that are in the receiving `compute` block.

That is, we check if 
- `I4 <= I1 + x * I3 + dx < I5` for some `x` in the general filtered solution of `x`.
and similarly for `y`.

If all constraints are satisfied, we have found a valid receiving block.
We match the send statement `S1` with the k-th `receive` statement `S2` in the local order
of the receiving block. If no such `receive` statement exists, we have a deadlock.

Next, we construct the predicate `P1` that describes which PEs in the sending
block send to the receiving block, which is given by the range and congruence constraints.
We may simplify the range constraints, as one of the two inequalities is trivially true
depending on if `dx` is positive or negative (and similarly for `dy`). Moreover, the congruence constraints
may be left out if `I6 = 1` or `J6 = 1`.

We add parametric stream edge `S1, (i, j)` to `S2, (i+dx, j+dy)` predicated by `P1`
to the list of stream edges. 

The algorithm takes `O(n^2)` time overall, where `n` is the number of `compute` blocks.
One can speed up the algorithm by filtering out all `compute` blocks that are not in the range of the stream
before solving the congruence relations. This can be done efficiently using 2D box intersection tests.

Note that the algorithm also checks for deadlocks by ensuring that all `send`s are matched with a `receive`.
To check if all `receive`s are matched with a `send`, we can use a similar algorithm
with the roles of `send` and `receive` reversed and using `dx` and `dy` negated.
Once we have the stream edges, we can check if the sizes of the stream edges are consistent
between sends and receives.

## *Parametric* Happens-Before Graph

We now define a parametric happens-before graph that describes the happens-before relations compactly.
The vertices in the graph are pairs of statements and pairs of symbolic PE coordinates in the variables `i` and `j`.
The symbolic expressions are restricted to be either constants or of the form `i + c`/`j+c` for some constant `c`.
The edges in the graph are associated with a predicate over `i` and `j`.
This predicate may also involve parameters of the kernel and constants.
We allow for range constraints and congruence constraints as in the parametric stream edges.
The meaning of the edge `S1, (i1, j1) -> S2, (i2, j2)` with predicate `P1` is that
if the predicate `P1` is true for some `i1`, `j1` in the compute block of `S1`, then
`S1` must complete at PE `(i1, j1)` before `S2` can start at PE `(i2, j2)`.

For example, `S1, (i, j)`, `S1, (i-1, 0)`, and `S2, (1, 0)` could be vertices in the graph.
Then, `S1, (i, j) -> S2, (i+1, j)` could be an edge in the graph,
and it might have the predicate `i < I-1` where `I` is the size of the PE grid in the `i` dimension.
Another edge could be `S1, (i, j) -> S2, (i, j)` with the predicate `i == I-1`.

We can construct the parametric happens-before graph for each phase with the following steps,
which follow the same rules as the formal happens-before graph.
Throughout, if a vertex or edge already exists, we do not add it again.

1. For each pair of statements `S1`, `S2` in the same compute block for which 
`S1 --> S2` in [local order](#constructing-the-local-order), add:
   - the vertices `S1, (i, j)` and `S2, (i, j)` to the graph.
   - the edge `S1, (i, j) -> S2, (i, j)` with the predicate `true`.
2. For each parametric stream edge `(S1, (i, j)) -> (S2, (i+dx, j+dy))` predicated by `P1`:
   - Add the vertices `S1, (i, j)` and `S2, (i+dx, j+dy)` to the graph.
   - Add the edge `S1, (i, j) -> S2, (i+dx, j+dy)` with the predicate `P1`.
3. For each parametric stream edge `S3, (i, j) -> S4, (i+dx, j+dy)` predicated by `P1`:
   - Consider all vertices `S1, (i, j)` in the graph for which `S1, (i, j) -> S3, (i, j)` with predicate `P2`
   and all vertices `S2, (i+dx, j+dy)` for which `S4 -> S2`.
   - Add an edge `S1, (i, j) -> S2, (i+dx, j+dy)` with the predicate `P1 && P2`.
4. Apply transitivity until convergence:
   - If there is an edge from `S1, (i, j)` to `S2, (i+dx, j+dy)` with predicate `P1`
   - and an edge from `S2, (i+dx, j+dy)` to `S3, (i+dx', j+dy')` with predicate `P2`,
   - then add an edge from `S1, (i, j)` to `S3, (i+dx', j+dy')` with predicate `P1 && P2`.

Whenever creating a new predicate from two predicates, we simplify the predicate
as much as possible. The resulting predicate remains a conjunction of range constraints
and congruence constraints.

[TODO: Show that there is a unique canonical representation for P1 && P2]

[TODO: Efficient implementation details]

### Analysis

The runtime is polynomial in the number of compute blocks and the number of stream edges
in a phase. [TODO: exact cost analysis]


## *Parametric* Routing Graph

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

For each stream `F`, go over all `send` statements `S1` in the local order.
Let `F` have `hops = [(dx_1, dy_1), ..., (dx_h, dy_h)]`
and let `v` be the compute block of `S1`.

We iteratively add edges as follows, the idea is to explore an implicitly defined
graph using DFS:
Initialize a stack of vertex-index pairs to visit and a set of visited vertex-index pairs.
Add the node-index pair `(v, 1)` to the stack.

Until the stack is empty:
Pop the top vertex-index pair `(u, k)` from the stack.
Consider the current vertex `u=[I1:I2:I3, J1:J2:J3]` and the next hop `(dx_k, dy_k)` at index `k`.
Add an edge `(u, w)` to each of the vertices `w` described hereafter,
labeling it with `F`, `v`, and `k`.
If `k == h`, pop the next `receive` statement `S2` from the stack of `w` and record the **stream edge** `(S1, S2)`.
Else if `(w, k+1)` is not in the visited set, add `(w, k+1)` to the stack.

**Case: The stride is `I3 == J3 == 1`:**

- To block `[I1:I2:1, J1:J2:1]` with predicate (the cases are mutually exclusive by definition because `|dx_k|+|dy_k|==1`):
  - `i + 1 < I1 - 1` if `dx_k == 1`
  - `i - 1 > I1` if `dx_k == -1`
  - `j + 1 < J1 - 1` if `dy_k == 1`
  - `j - 1 > J1` if `dy_k == -1`

- If `dx_k != 0`, to all blocks `[I4:I5:1, J4:J5:1]` for which `J4 < J2`, `J5 >= J1`, and
  - for which `I4 == I2 + 1` with the predicate `i = I2 && J4 <= j < J6` if `dx_k == 1`
  - for which `I5 == I1 - 1` with the predicate `i = I1 && J4 <= j < J6` if `dx_k == -1`
  - Note that the ranges `J4:J5` of all such blocks must together cover the range `J1:J2`.
Failure to do so constitutes an incorrect declaration of stream edges (deadlock).

- Proceed symmetrically in case `dy_k != 0`.


**Case: All compute blocks have the same stride > 1:**

- If `dx_k != 0`, to all blocks `[I4:I5:I3, J4:J5:J3]` for which `J4 < J2`, `J5 >= J1`, and 
for which `I4 = I2 + dx_k (mod I3)` with the predicate `I4 <= i + 1 < I5 && J4 <= j < J6`.
  - Note that the ranges `J4:J5` of all such blocks must together cover the range `J1:J2`, and similarly, 
the ranges `I4:I5` must together cover the range `I1+dx_k:I2+dx_k`.
Failure to do so constitutes an incorrect declaration of stream edges (deadlock).

- Proceed symmetrically in case `dy_k != 0`.

**General Case**

In the general case, we can use the same constraints & method we used to compute stream edges
for the current hop `(dx_k, dy_k)`, it uses congruence relations to determine the receiving blocks.

*Note, I did the special cases first, so I kept them for now.
We can also use the general algorithm for all cases,
but we should make sure to be able to simplify all the predicates in the special cases.*

### Analysis

Note that blocks with a single element can be interpreted as having any arbitrary stride.
This is useful for implementing boundary conditions.

Runtime: Note that each vertex is added to the stack at most `h` times, where `h` is the number of hops.
Adding all edges for a given vertex takes at most `n` time,
where `n` is the number of `compute` blocks.
Hence, the overall runtime is `O(n^2 * h)`.
The space complexity is `O(n * h)`.

## The Conflict Graph

The conflict graph can be used to determine if a routing declaration is correct,
and resolve the `auto` routing declarations.
The conflict graph is a directed graph that describes the conflicts between streams.
Two streams conflict if they are routed through the same channel at some shared PE
and are not ordered by happens-before.

We use the parametric routing graph to construct the conflict graph.



