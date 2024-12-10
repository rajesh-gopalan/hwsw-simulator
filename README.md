# hwsw-simulator
The Hardware-Software Co-design Simulator aims to model the interaction between hardware components and software workloads, allowing for performance analysis and optimization strategies.


Hardware Model
• Implement simplified models of key hardware components:
  ◊ CPU (with multiple cores)
  ◊ GPU
  ◊ Memory hierarchy (L1, L2, L3 caches, main memory)
  ◊ Storage (SSD and HDD)
• Allow for configurable parameters such as clock speeds, cache sizes, and memory bandwidths

Workload Generator
• Create synthetic workloads that represent typical applications:
  ◊ Compute-intensive tasks
  ◊ Memory-bound operations
  ◊ I/O-heavy processes
  ◊ Implement a simple instruction set to represent these workloads

Simulation Engine
• Develop a discrete event simulation engine
• Model how instructions flow through the hardware components
• Track resource utilization and contention

Performance Metrics
• Implement key performance indicators:
  ◊ Execution time
  ◊ Throughput
  ◊ Resource utilization (CPU, GPU, memory, storage)
  ◊ Power consumption estimation

Visualization and Analysis
• Create graphs and charts to visualize:
  ◊ Resource utilization over time
  ◊ Performance bottlenecks
  ◊ Comparative analysis of different hardware configurations

