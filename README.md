# Options-for-accelerating-routing-using-CUDAData-
This repository contains high-performance CUDA C++ implementations for IP routing table lookups using the Longest Prefix Match (LPM) algorithm. The project aims to evaluate and achieve 800G line-rate compliance (approximately 1190 Million packets per second - Mpps) by leveraging GPU acceleration. 

The implementations compare various GPU memory architectures (Global Memory, Texture Cache, Shared Memory) against a multi-threaded CPU reference implementation (using OpenMP).

## Features

*   **IPv4 LPM Search:** Implements a 16-8-8 bit stride Trie architecture.
*   **IPv6 LPM Search:** Implements both a deep Single-Bit Trie and an optimized 11-8-8-8-8-8 bit stride Trie using GPU Shared Memory.
*   **Memory Architectures Tested:** Global Memory, Texture Objects (Texture Cache), and Shared Memory.
*   **Asynchronous Execution:** Utilizes CUDA Streams for efficient data transfer and kernel execution pipelining.
*   **CPU Baseline:** Includes OpenMP-accelerated CPU reference functions for direct performance and energy efficiency (Mpk/J) comparisons.
*   **Automated Benchmarking:** Built-in sweep tests for batch sizes and thread scaling, automatically logging results to CSV files.

## System Requirements

To achieve the exact performance metrics and run the benchmarks as intended, your system should match or exceed the following specifications.

### Minimum Hardware Requirements (Baseline Setup)
*   **CPU:** AMD Ryzen 7 3700X (8 Cores / 16 Threads) or equivalent. *(Required for accurate 16-thread CPU baseline comparisons).*
*   **GPU:** NVIDIA GeForce RTX 2070 SUPER (Turing architecture, 8GB VRAM) or equivalent/better.
*   **RAM:** 16 GB DDR4 (To safely handle large pinned memory allocations and multi-million packet batch sizes).
*   **PCIe Interface:** PCIe Gen 3.0 x16 or higher (Crucial for Host-to-Device transfer speeds).

### Software Prerequisites
*   **Operating System:** Windows 10/11 (The current IPv6 implementation relies on `<winsock2.h>` for IP parsing). *For Linux, minor code adaptations replacing Winsock with `<arpa/inet.h>` are required.*
*   **CUDA Toolkit:** Version 11.0 or higher (Tested with version 13.1+).
*   **C++ Compiler:** MSVC (Microsoft Visual C++) included with Visual Studio 2019/2022.
*   **OpenMP:** Supported natively by MSVC.

## Setup & Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/ErikJakubicka/Options-for-accelerating-routing-using-CUDAData-](https://github.com/ErikJakubicka/Options-for-accelerating-routing-using-CUDAData-)
cd Options-for-accelerating-routing-using-CUDAData-
```

**2. Verify CUDA Installation:**
Ensure that the `nvcc` compiler is available in your system path by running:
```bash
nvcc --version
```

## Data Preparation

The programs require BGP routing table dumps to build the search Tries. You must place these text files in the root directory alongside the executables.

1.  **IPv4 Prefixes:** Create a file named `unique_prefixes.txt`.
    *   *Format:* `IP/Mask` (e.g., `192.168.1.0/24`)
2.  **IPv6 Prefixes:** Create a file named `ipv6_prefixes.txt`.
    *   *Format:* `IP/Mask` (e.g., `2001:db8::/32`)

## Compilation Instructions

Use the NVIDIA CUDA Compiler (`nvcc`) to build the source files. Open your command prompt (preferably the "x64 Native Tools Command Prompt for VS") and run the following commands. Note the `-std=c++17` flag which is required for modern Thrust library compatibility.

*(Make sure to adjust the `.cu` filenames if you saved them differently).*

**Compile IPv4 Global Memory implementation:**
```bash
nvcc -O3 -std=c++17 -Xcompiler "/openmp" IPv4_16-8-8_s_global.cu -o ipv4_global.exe
```

**Compile IPv4 Texture Cache implementation:**
```bash
nvcc -O3 -std=c++17 -Xcompiler "/openmp" IPv4_16-8-8_s_texture.cu -o ipv4_texture.exe
```

**Compile IPv4 Single-Bit implementation:**
```bash
nvcc -O3 -std=c++17 -Xcompiler "/openmp" IPv4_single_bit.cu -o ipv4_single_bit.exe
```

**Compile IPv6 Shared Memory implementation:**
```bash
nvcc -O3 -std=c++17 -Xcompiler "/openmp" IPv6_s_shared_memory.cu -lws2_32 -o ipv6_shared_memory.exe
```

**Compile IPv6 Single-Bit implementation:**
```bash
nvcc -O3 -std=c++17 -Xcompiler "/openmp" IPv6_single_bit.cu -lws2_32 -o ipv6_single_bit.exe
```

*Flag Explanations:*
*   `-O3`: Applies maximum compiler optimizations.
*   `-std=c++17`: Compiles using the C++17 standard (Required for newer versions of CUDA's Thrust library).
*   `-Xcompiler "/openmp"`: Enables OpenMP support for the CPU reference tests on Windows MSVC.
*   `-lws2_32`: Links the Windows Sockets library required for parsing IPv6 addresses.

## Running the Benchmarks

Once compiled, simply execute the generated binaries. Ensure no heavy background tasks are running to get accurate latency and throughput measurements.

```bash
.\ipv4_global.exe
.\ipv4_texture.exe
.\ipv4_single_bit.exe
.\ipv6_shared_memory.exe
.\ipv6_single_bit.exe
```

### What to expect during execution:
1.  **Trie Building:** The program will parse the prefix files and build the search tree, displaying its memory footprint.
2.  **Sweep Test:** Runs GPU throughput tests across various packet batch sizes (from 1,024 to 10,000,000) to find the optimal saturation point.
3.  **Stream Test:** Evaluates pipelining efficiency using 2, 3, and 4 concurrent CUDA streams.
4.  **CPU Scalability:** Tests the OpenMP CPU implementation from 1 to 16 threads.
5.  **End-to-End Stress Test:** Compares the best GPU configuration against the maximum CPU performance, calculating energy efficiency (Mpk/J) and total speedup.

## Outputs and Results

All benchmark metrics are printed to the console in real-time. Additionally, the programs automatically generate the following **CSV files** for easy importing into Excel/Python for graphing:

*   `test_sweep.csv` - Data from batch size scaling.
*   `test_streams.csv` - Data from CUDA stream overlapping efficiency.
*   `test_cpu.csv` - Data from CPU thread scaling.
*   `test_final.csv` - The final comparison summary between GPU and CPU.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
