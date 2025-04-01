# **Parallel 2D Convolution using CUDA**  
### **Optimized Image Processing | C/C++ | CUDA**  

---

## **📖 Overview**  
This project implements **parallelization of 2D convolution** using **CUDA** for execution on an NVIDIA GPU.
It is an extension of HW3, where convolution was parallelized using **OpenMP**.

In this implementation, three different CUDA-based convolution approaches are explored:  
1. **Naive Parallel Convolution** – Each thread processes a single output pixel using global memory.  
2. **Constant Memory Optimized Convolution** – The kernel is stored in constant memory to reduce memory access latency.  
3. **Tiled Shared Memory Convolution** – Shared memory is used to optimize data locality and reduce redundant memory accesses.  

Performance is analyzed using **Nvprof** to compare execution times for three different implementations.  

🔹 **util.cpp**, provided by the TA, handling memory allocation, file reading, and other utilities, while also using the edge-extending code from HW3.

🔹 The **"maximum number of threads per block (1024)"** is selected based on the NVIDIA GPU query in Google Colab.

---

## **📂 Directory Layout**  
```bash
📦 CODE
├── 📂 data/                  # Input datasets
│   ├── image_large.txt         # test/input image
│   ├── tiny_kernel.txt         # test kernel
│   ├── tiny_kernel2.txt        # test kernel
│   ├── tiny_text.txt           # test image
├── 📂 src/                   # Source code
│   ├── conv_naive.cu           # Basic convolution
│   ├── conv_cons.cu            # Optimized convolution (constant memory)
│   ├── conv_tiled.cu           # Optimized convolution (shared memory & tiling)
│   ├── util.cpp                # Helper functions
│   ├── util.h                  # Utility header file
├── 📂 bin/                   # Compiled binaries
├── 📂 results/               # Output files
│   ├── naive_output.txt        # Output from naive convolution
│   ├── cons_output.txt         # Output from constant memory convolution
│   ├── tiled_output.txt        # Output from tiled convolution
├── 📜 Makefile               # Build script
├── 📜 text_to_jpg.py         # Converts output text to an image
```

---

## **⚡ Compilation & Execution**  
### **1️⃣ Prerequisites**  
NVIDIA GPU with CUDA support.\
Ensure that the NVIDIA's CUDA compiler is installed (nvcc is part of the NVIDIA CUDA Toolkit).\
You can check by running:
```bash
nvcc --version
```

Python 3 and the Pillow library for image conversion:
```bash
pip install pillow
```

## **2️⃣ Compilation**
Navigate to the project directory and use:
```bash
make
```
This will compile:
- conv_naive → Basic CUDA convolution

- conv_cons → Optimized version using constant memory

- conv_tiled → Optimized version using shared memory & tiling

## **3️⃣ Running the Implementations**
▶ Running the Naive CUDA Convolution
```bash
make run_naive
```

```bash
make run_cons
```

```bash
make run_tiled
```

Equivalent to:
```bash
./bin/run_{method} data/input.txt data/tiny_kernel2.txt results/naive_output.txt
```
```bash
python3 text_to_jpg.py results/{method}_output.txt
```

**Ex:**  
*tiny_kernel2.txt* (convolution kernel) is applied to *image_large.txt* (input image), and the output is written as *{method}_output.txt*."

Converts the result into an image using text_to_jpg.py


---

To convert an input image to a png file:

```bash
make run_default_image
```

## **4️⃣ Profiling with Nvprof**
To analyze performance using Nvprof:

```bash
nvprof ./bin/conv_naive data/input.txt data/tiny_kernel2.txt results/naive_output.txt
nvprof ./bin/conv_cons data/input.txt data/tiny_kernel2.txt results/cons_output.txt
nvprof ./bin/conv_tiled data/input.txt data/tiny_kernel2.txt results/tiled_output.txt
```

## **5️⃣ Cleaning Up Build & Results**  
```bash
make clean
```
---  
