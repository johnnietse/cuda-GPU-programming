# cuda-GPU-programming

## Overview
This repository contains my work for Machine Problem 1 (MP1) and Machine Problem 2 (MP2) from the Digital Systems Engineering course at Queen's University. These assignments focus on GPU programming using CUDA to explore matrix multiplication and device capabilities on NVIDIA GPUs.

## Machine Problem Details
### MP1: Device Query and Basic Matrix Multiplication
- Part 1: Device query to identify GPU capabilities (clock rate, memory, threads, etc.)
- Part 2: Implementation of a basic GPU matrix multiplication kernel with performance analysis
  
    - Data transfer time measurements (host-to-device and device-to-host)
    - CPU vs GPU performance comparison
    - Thread/block configuration experiments

### MP2: Optimized Tiled Matrix Multiplication
- Part 1: Implementation of a shared memory (tiled) matrix multiplication kernel
  
    - Performance analysis with different tile sizes (2, 4, 8, 16, 32)
    - Comparison with MP1 baseline implementation

- Part 2 (Bonus): Revised tiled kernel with boundary checks for non-square matrices

## GPU Environment
All work was completed on one of the following GPU machines:
  - 24 remote access machines with NVIDIA T600

All machines run:
  - Visual Studio 2022
  - CUDA 12.2

## Repository Structure
ELEC374-Machine-Problems/
│
├── MP1/
│   ├── MP1_Part1.cu       # Device query implementation
│   ├── MP1_Part2.cu       # Basic matrix multiplication
│   └── MP1_Report.pdf     # Analysis and results
│
├── MP2/
│   ├── MP2_Part1.cu       # Tiled matrix multiplication
│   ├── MP2_Part2.cu       # Boundary-checked version (bonus)
│   └── MP2_Report.pdf     # Analysis and results
│
└── README.md              # This file
