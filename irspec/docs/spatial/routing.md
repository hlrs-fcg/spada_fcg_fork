# Semantics of Routing Declarations

Routing declarations must respect the limitations on how `channel`s are used.
Specifically, it must be avoided that two messages are routed through the same `channel` 
at the same PE simultaneously.

## The Routing Graph

The routing graph of a phase is a directed graph that describes how data is routed between PEs.
Note that the routing graph is defined in terms of the PE coordinates, so
its size grows with the size of the PE grid. It serves as a formal model for defining
the semantics, but should not be constructed explicitly.

Recall that stream edges are pairs of [send](../spatial#streaming-data-with-send) 
and [receive](../spatial#receiving-streaming-data-with-receive) operations that are matched across PEs.
Stream edges must not cross [phases](../spatial#phases), that is, a stream edge must be entirely contained within a phase.

The routing graph contains the following nodes *V*, edges *E*, and paths *P*:

- Each PE is a node in the graph.
- Consider each stream edge from PE `(x_1, y_1)` to PE `(x_2, x_2)` going through stream *F* on channel *C* through PE `hops = [(dx_1, dy_1), (dx_2, dy_2), ..., (dx_n, dy_n)]`.
We add an edge from `(x_1+dx_i, y_1+dy_i)` to `(x_1+dx_{i+1}, y_1+dy_{i+1})` for each *i* in *0, ..., n*.
where we use the convention that `dx_0 = dy_0 = 0`.
- Moreover, we add the resulting path `(x_1, y_1), ..., (x_1+dx_i, y_1+dy_i), ..., (x_2, y_2)` to the list of paths *P*
and record the stream *F* and channel *C*.


??? example "Example: 2-phase Reduce"
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

## Undefined Behavior

Next, we describe the condition under which the routing behavior is undefined:

We say that *P1* happens-before *P2* and write `P1 -> P2` if
the `receive` of *P1* happens-before the `send` of *P2*.

**If two paths *P1* and *P2* in the routing graph of a phase using the same channel
that share a PE `(x, y)` and *P1* and *P2* are not ordered by happens-before,
then the behavior is undefined.**

This is because the two messages may interfere with each other
and the order in which they are processed may become nondeterministic.
Recall that sending onto the same stream [must be synchronized using completions
to avoid data races](../spatial#streaming-data-with-send). Hence, sending through the same stream multiple times
in the same phases is ok as long as the sends (and receives) are correctly synchronized.

Keep in mind that PEs transition between phases asynchronously,
that is, a PE may advance to the next phase before another PE has completed the current phase.
We exploit here implicitly that routers back-pressure when
they receive data from a channel on which they are not configured
to receive. 

*A Note regarding potential extensions.*
The current definition is tailored to the case
where all streams are point-to-point paths.
If multicasting is used, the correctness conditions become more challenging
to specify.