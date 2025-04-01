# **HW3: Parallel Image Processing using OpenMP**
### Blurring, Denoising, and Extension | C/C++ | OpenMP

---  

## **📖 Overview**
This project implements parallel **image processing** techniques—including **blurring**, **edge-extending**, and **denoising** using OpenMP for parallel execution in C++. Both sequential and parallel versions are implemented to analyze performance improvements across multiple threads. The same image and kernel are applied across tests for performance evaluation.


A **Python utility** (`text_to_jpg.py`) is provided to visualize text-based image representations by converting them into **JPG format**.  

---  

🔹 **The overall structure of the code follows the homework framework provided by the TA, with `util.cpp` handling memory allocation, file reading, and other utilities.**  

## **📂 Directory Layout**
```bash
📦 Code
├── 📜 compile_and_run_test.sh     # Compilation & execution script
├── 📂 example-image-kernel/       # Example images & convolution kernels
│   ├── image_large.txt              # Large sample image (text format)
│   ├── kernel_large.txt             # Large kernel
│   ├── kernel.txt                   # Standard kernel
│   ├── tiny_kernel2.txt             # Small test kernel
│   ├── tiny_kernel.txt              # Small test kernel
│   ├── tiny_text.txt                # Small test image
├── 📜 filtering_seq.cpp           # Sequential image filtering
├── 📜 filtering_omp.cpp           # OpenMP parallel image filtering
├── 📜 util.cpp                    # Provided in HW. Utility functions (file I/O, memory management)
├── 📜 util.h                      # Header file for utilities
├── 📜 text_to_jpg.py              # Converts text-based images to JPG
```

## **⚡ Compilation & Execution**  

### **1️⃣ Prerequisites**  
Ensure that your system has:  
- A **C++ compiler** with **OpenMP support** (GCC with `-fopenmp`).  
- **Python 3** and the **Pillow** library for image conversion. Install Pillow with:  
```bash
  pip install pillow
```

### **2️⃣ Compilation and Running**
Run the provided script to compile the sequential and OpenMP versions:

```bash
source compile_and_run_test.sh
```
This script compiles:

filtering_seq → Sequential filtering\
filtering_omp → Parallel filtering (OpenMP)

### **3️⃣ Measuring Performance**
Execution time is measured using:\
🔹 omp_get_wtime() (inside the code) \
🔹 time command (external measurement for real, user, and system time)

Both timing results are printed in the output for comparison.\

### **4️⃣ Converting Output to JPG**
This will generate test image and blurred image in the working directory.

```bash
python3 text_to_jpg.py $testimage
python3 text_to_jpg.py $blurredOutput
```
