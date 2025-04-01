all: quicksort_serial hypercube_quicksort_parallel

CFLAGS = -Wall -Wextra -Wpedantic -I src

# serial
quicksort_serial: src/quicksort_serial.c src/custom_vector.c
	mkdir -p bin
	gcc $(CFLAGS) src/quicksort_serial.c src/custom_vector.c -o bin/quicksort_serial

# parallel
hypercube_quicksort_parallel: src/hypercube_quicksort_parallel.c src/custom_vector.c
	mkdir -p bin
	mpicc $(CFLAGS) src/hypercube_quicksort_parallel.c src/custom_vector.c -o bin/hypercube_quicksort_parallel -lm

clean:
	rm -rf bin/* results/*

runs: quicksort_serial
	mkdir -p results
	./bin/quicksort_serial data/input.txt results/sorted_serial.txt

runp: hypercube_quicksort_parallel
	mkdir -p results
	mpirun -np 4 ./bin/hypercube_quicksort_parallel data/input.txt results/sorted_parallel.txt
