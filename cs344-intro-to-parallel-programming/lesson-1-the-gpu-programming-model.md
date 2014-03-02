# Lesson 1: The GPU Programming Model

* "How to dig a hole faster"
    * dig faster
    * buy more productive shovel
    * hire more diggers
* Parallel: methods for building faster processor
    * Run with a faster clock
        * Shorter amount of time on each step of computation
    * Do more work per clock cycle
    * However, power comsumption of chip is at the limit
    * Add more processes
* Modern GPU
    * thousands of ALUs (arithmetic units)
    * hundreds of processors
    * tens of thousands of concurrent threads
* CPU speed remaining flat
    * feature size of processor == minimal size of transistor or wire on a chip 
    * as feature size gets smallers, transistors get
        * smaller
        * faster
        * use less power
    <img src="./images/feature_size_graph.png"></img>
    * we've generally stopped increasing clock speed
        * mostly because of heat and power related limitations
    <img src="./images/clock_rate_stagnant.png"></img>
    * Instead:
        * more and smaller processors
* Optimisation types
    * Latency (time) - CPUs optimise for this
    * Throughput (stuff/time) - GPUs optimise for this
        * Examples: Pixels matched per second
* CUDA Program Diagram
    * Computers are "heterogenous": they have two different processors in them
    * Host: part of program runs on CPU
    * Device: part of program that runs on GPU - extensions for parallelism
    * Assume host and device have separate memories to store data (DRAM)
    * *CPU is in charge!* Does the following:
        1. Move data from CPU memory to GPU
        2. Move data from GPU back to CPU
            * ```cudaMemcpy``` is the command to do this
        3. Allocate GPU memory
            * ```cudaMalloc```
        4. Launch kernel on GPU
        <img src="./images/cuda.png"></img>
* Big Idea of GPU Computation
    * When you write a kernel (a function to run on GPU), write as if a single thread, then GPU will run on many threads
* GPU is good at:
    * launching lots of threads
    * running lots of threads in parallel
