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


Thoughts about lowering:


- Tasks
  - We create a dependency graph between statements
    - This graph defines the happens-before partial order
  - This task-graph can be converted into a list of tasks
    - and associated blocks/unblocks
- Routing
  - We create the parametric graph templates from the dataflow edges,
  - Then, we construct the conflict graph also using the happens-before relation
  - A coloring of the conflict graph gives us the channel assignments
  - The routing table is then constructed from the channel assignments
  - If the coloring fails to provide a small enough number of colors,
    - we can split the computation into multiple parts and schedule them at different times
    - this requires a (global) reconfiguration mechanism