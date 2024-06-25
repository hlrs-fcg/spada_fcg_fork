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


```
kernel vadv(f32[I, J, K] utens_stage,
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
      f32[K] utens_stage[i, j, 0:K];
      f32[K] u_stage[i, j, 0:K];
      f32[K] wcon[i, j, 0:K];
      f32[K] u_pos[i, j, 0:K];
      f32[K] utens[i, j, 0:K];
      f32[K] datacol[i, j, 0:K];
  }
  
  
  ////
  // Computation and communication
  
  // Set up communication streams
  // The only communication is for wcon, which is sent to the west
  
  stream westwards = relative_stream(-1, 0);
  
  map spatial i, j in [0, 0:J] {
    // Boundary condition
    // ...
    on_receive(westwards, K) -> k, x {
      // ...
    }
  }
  
  map spatial i, j in [1:I, 0:J] {
  
      # Read the input fields to local fields (memcopy)
      f32[K] wcon_l <- wcon;
      f32[K] ustage_l <- u_stage;
      f32[K] u_pos_l <- u_pos;
      f32[K] utens_l <- utens;
      f32[K] utens_stage_l <- utens_stage;
      
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
            
            utens_stage_l -> utens_stage;
            datacol_l -> datacol;
        }
        
      }   
  }

}


```


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
```