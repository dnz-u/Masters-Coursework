# **Hypercube Quicksort Implementation**  
### Parallel Quicksort | C | MPI

---  

## **📖 Overview**  
This project implements **Quicksort** in both **serial** and **parallel** versions. The parallel implementation uses **MPI (Message Passing Interface)** and a **hypercube topology** to distribute and sort data across multiple processors.  

---  

## **📂 Directory Layout**  
```bash
📦 code
├── 📜 Makefile         # Build script
├── 📂 src/             # Source code
│   ├── quicksort_serial.c               # Serial implementation
│   ├── hypercube_quicksort_parallel.c   # MPI-based parallel implementation
│   ├── custom_vector.c                  # Dynamic array implementation
│   └── custom_vector.h                  # Header file
├── 📂 bin/             # Compiled binaries
├── 📂 data/            # Input datasets
│   ├── input.txt         # Sample input data
├── 📂 results/         # Output files
│   ├── sorted_serial.txt                # Serial sorted output
│   ├── sorted_parallel.txt              # Parallel sorted output
│   ├── sorted_parallel.txt_part_{rank}  # Partial sorted output per processor
```  

---  

## **⚡ Compilation & Execution**  

### **1️⃣ Prerequisites**  
Ensure that the **MPI library** is installed. You can check by running:  
```bash
mpicc --version
```

### **2️⃣ Compilation**  
Navigate to the project directory and use:  
```bash
make
```
This will compile both:  
- `quicksort_serial` (Serial Quicksort)  
- `hypercube_quicksort_parallel` (Parallel Quicksort using MPI)  

### **3️⃣ Running the Serial Version**  
```bash
make runs
```
Equivalent to:  
```bash
./quicksort_serial input.txt sorted_serial.txt
```
- Reads input from **input.txt**  
- Outputs the sorted array to **sorted_serial.txt**  

### **4️⃣ Running the Parallel Version (MPI)**  
```bash
make runp
```
Equivalent to:  
```bash
mpirun -np <num_processes> ./hypercube_quicksort_parallel input.txt sorted_parallel.txt
```
- Reads input from **input.txt**  
- Outputs the final sorted array to **sorted_parallel.txt**  
- Each processor also prints its sorted chunk to **sorted_parallel.txt_part_{processor_rank}.txt**  

### **4️⃣ Cleaning Up Build & Results**  
```bash
make clean
```
---  
