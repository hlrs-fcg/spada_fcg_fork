# Virtual-Channel Routed IR

The goal is to serve as an intermediate representation
between spatial IR and CSL or a similar language.

Goals:

* Represent virtual channels and their routing
  * Each PE has a set of virtual channels (limited resource)
  * Each PE has a routing table for each channel
  * The network is a 2D mesh
  * The network can switch between routing configurations
    * This can be done locally triggered by a task
    * This can be done triggered by a message
    * A small number of configurations can be kept in 'router registers'
    * Globally blocking reconfiguration can be done using a collective operation
      * This rewrites the router registers and blocks communication for the duration (on that color?)
* Represent tasks and their dependencies
  * Tasks have IDs (limited resource)
  * Tasks may trigger on arrival of data
  * Task may block/trigger other tasks
  * Tasks may be asynchronous
* Represent vectorized operations (through the use of DSD-like ops)
  * Affine array accesses can be represented through vectorized-streaming instructions
  * Memory movement from the network to the PE can be represented through such vectorized move instructions
  * Vectorized instructions may be asynchronous, and their completion may trigger tasks
  * Management of DSRs (limited resource)
* Represent host-to-PE communication
    * Host can send data to PE
    * Host can trigger tasks on PE (remote procedure calls)
    * Host can receive data from PE
    * This can be done in a streaming fashion or in a blocking fashion (copy semantics)
* Manage and represent PE memory
  * Manage PE memory (limited resource)
  * Both program size and data size contribute to memory usage
* Represent Fabric<->PE Queues?
  * Manage queues between the fabric and the PE (limited resource)
* Management of limited resources
  * The compiler should optimize the use of limited resources
  * If a resource is full, the system should be able to handle this.
    For example, by splitting computation into multiple parts and scheduling them at different times.
    Or by blocking tasks until resources are available.

What do we abstract away compared to CSL?:

* How to switch between routing configurations
* How to manage DSRs
* How to manage task IDs

Thoughts about lowering:

- Tasks
  - We create a dependency graph between statements
    - This graph defines the happens-before partial order
  - This task-graph can be converted into a list of tasks
    - and associated blocks/unblocks
- Routing
  - We create the parametric graph templates from the dataflow edges,
  - Turn point-to-point messages into actual paths (routing - sssp)
  - Then, we construct the conflict graph also using the happens-before relation
  - A coloring of the conflict graph gives us the channel assignments
  - The routing table is then constructed from the channel assignments
  - If the coloring fails to provide a small enough number of colors,
    - we can split the computation into multiple parts and schedule them at different times
    - this requires a (global) reconfiguration mechanism
    - can happen around stream edges (either before send/receive whichever is 'first')

TODO: Add barriers to level 1 IR

## PE IDs

Each PE has a unique cartesian ID `(i, j)`. This ID is used to identify the PE in the network.

## Virtual Channels


### Virtual Channel Declaration

To declare a virtual PE-PE channel, we use the following syntax:

```rust
// A virtual channel declaration
pe_channel<T> channel_name;
```

### Host-to-PE Communication

TODO Declare memcpy interface.

### Routing Tables
To configure a virtual channel, we need to specify the routing table for each PE.

For example, 
```rust
// A routing table for a virtual channel
routing_table name {
  receive: {EAST}, 
  send: {WEST, RAMP}
};
```

### Setting Routing Tables for Virtual Channels

To set the routing table for a virtual channel, we use the following syntax:

```rust
// Set the routing table for a virtual channel
set_routing_table(pe_channel_name, routing_table_name);
```

The representation abstract away the details of how configurations are changed.
Depending on the context, this can be done differently.
For example, this might require a switch-advance, teardown, or global reconfiguration using a barrier.

It may be assumed that when the call is made, the configuration of the calling PE has been updated.

**Sending or receiving messages on a channel while concurrently updating the configuration constitutes undefined behavior.**

## Tasks

A task is a unit of computation that can be triggered by the arrival of data or by another task.

```rust
// a local task is triggered by other tasks
task taskname() {
    // Statements
}

// a data task is triggered by the arrival of data
task taskname(T variable_name) {
    // Statements
}
```
where `T` is the (scalar) type of the data that triggers the task.

Tasks can be active or inactive, blocked or unblocked.
A task can run when it is active and unblocked.
Initially, all tasks ar inactive and unblocked.

To change the state of a task, we use the following syntax:
```rust
// Block a task
block_task(taskname);

// Unblock a task
unblock_task(taskname);

// Activate a local task
// Data tasks cannot be activated directly
activate_task(taskname);
```
A data tasks becomes active when data arrives.

[TODO: Check if this makes sense!!]
A task is deactivated upon completion.

[TODO: Check if needed]

Activating a task concurrently multiple times is considered undefined behavior.

// TODO How to set the starting task?
// Is there a predefined main task?

## Vectorized Operations

TODO: How to define the DSDs?

TODO: This includes fabric operations (e.g., fabric-to-CE communication)

## PE Memory

TODO: Define fields, including scalars and arrays.

## Parameters

TODO: Define parameters, that are passed at compile time.


## Examples

```rust
// A simple laplacian 2D stencil



```