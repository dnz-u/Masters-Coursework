/*
@author: Deniz Uzel
*/

#include <cstdio>
#include "util.h"
#include <cmath>


#define MAX_NO_THREADS_IN_A_BLOCK 1024 // Tesla K4 GoogleColab
#define MAX_WIDTH 32

#define KERNEL_NUMBER_OF_ELEMENTS 100


__constant__ int d_kernel[KERNEL_NUMBER_OF_ELEMENTS];


__global__ void conv(int* d_in,  int* d_out,
                     int kernel_size,
                     int num_rows, int num_cols,
                     int extended_num_rows, int extended_num_cols,
                     int shared_memory_size)
{

    // amount of padding for each side
    int p = kernel_size / 2;

    // number of thread num in one side
    int BLOCK_WIDTH = MAX_WIDTH;

    // initialize tile array
    __shared__ int tile[MAX_NO_THREADS_IN_A_BLOCK];

    int inner_tile_width = BLOCK_WIDTH - 2*p;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row_out = blockIdx.y*inner_tile_width + ty;
    int col_out = blockIdx.x*inner_tile_width + tx;

    int row_start = row_out;
    int col_start = col_out;

    if ((row_start < extended_num_rows) && (col_start < extended_num_cols)
        && row_start >= 0 && col_start >=0){
        tile[ty*BLOCK_WIDTH + tx] = d_in[row_start*extended_num_cols + col_start];
    } else {
        tile[ty*BLOCK_WIDTH + tx] = 0;
    }


    __syncthreads();

    // checks if the thread is valid for the convolution
    if ((ty < inner_tile_width) && (tx < inner_tile_width)){

        // convolution output value
        int sum_val = 0;

        for (int i = ty; i < (ty+kernel_size); i++){
            for (int j = tx; j < (tx+kernel_size); j++){

                int val_in = tile[i*BLOCK_WIDTH + j];
                int val_ker = d_kernel[(i-ty)*kernel_size + j-tx];

                sum_val += val_in * val_ker;
            }
        }

        // normalize the output value
        int kernel_sum = 0;

        for (int i=0; i < kernel_size*kernel_size; i++){
                kernel_sum += d_kernel[i];
        }

        double normalized_val = (double)sum_val / kernel_sum;

        if (row_out < num_rows && col_out < num_cols){
            d_out[row_out*num_cols + col_out] = (int)normalized_val;
        }
    }
}


int main(int argc, char* argv[]) {

    int num_rows, num_cols, kernel_size;

    // load image and kernel
    int** img = read_pgm_file(argv[1], &num_rows, &num_cols);
    int** kernel = read_pgm_file(argv[2], &kernel_size, &kernel_size);

    // extend the image
    int* h_extendedImg;

    int extend_amount = kernel_size / 2;
    int extended_num_rows = num_rows + extend_amount*2;
    int extended_num_cols = num_cols + extend_amount*2;

    h_extendedImg = extend_edges(img, num_rows-1, num_cols-1, extend_amount);


    // CUDA Part

    const int INPUT_ARRAY_SIZE = extended_num_rows * extended_num_cols;
    const int INPUT_ARRAY_BYTES = INPUT_ARRAY_SIZE * sizeof(int);

    const int OUTPUT_ARRAY_SIZE = num_rows * num_cols;
    const int OUTPUT_ARRAY_BYTES = OUTPUT_ARRAY_SIZE * sizeof(int);

    const int KERNEL_ARRAY_SIZE = kernel_size * kernel_size;
    const int KERNEL_ARRAY_BYTES = KERNEL_ARRAY_SIZE * sizeof(int);


    int* h_kernel = convert_kernel_2d_to_1d(kernel, kernel_size, kernel_size);
    int* h_out = (int*)malloc(OUTPUT_ARRAY_BYTES);

    int* d_in;
    int* d_out;

    cudaMalloc((void**)&d_in, INPUT_ARRAY_BYTES);
    cudaMalloc((void**)&d_out, OUTPUT_ARRAY_BYTES);
    cudaMalloc((void**)&d_kernel, KERNEL_ARRAY_BYTES);

    cudaMemcpy(d_in, h_extendedImg, INPUT_ARRAY_BYTES, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_ARRAY_BYTES);

    // amount of padding for each side
    int p = kernel_size / 2;

    // dimensions
    int thread = MAX_WIDTH; // MAX_NO_THREADS_IN_A_BLOCK >> 5;  // 1024/32 = 32
    int block_width = ceil((double)extended_num_rows / (thread-2*p));  // +1 is for defensive coding
    int block_height = ceil((double)extended_num_cols / (thread-2*p));

    dim3 threads_per_block(thread, thread);
    dim3 blocks_per_grid(block_width, block_height);


    // Total amount of shared memory per block: 49152 bytes
    const int SHARED_MEMORY_SIZE = MAX_NO_THREADS_IN_A_BLOCK;
    const int SHARED_MEMORY_BYTES = SHARED_MEMORY_SIZE * sizeof(int);

    // To Do: Conv KERNEL HERE
    conv <<<blocks_per_grid, \
            threads_per_block, \
            SHARED_MEMORY_BYTES >>> (d_in, d_out,
                                     kernel_size,
                                     num_rows, num_cols,
                                     extended_num_rows, extended_num_cols,
                                     SHARED_MEMORY_SIZE);

    cudaMemcpy(h_out, d_out, OUTPUT_ARRAY_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);

    // write the output to file
    char* output_file_name = argv[3];
    writeFile_1d(output_file_name, h_out, num_rows, num_cols);

    free(h_extendedImg);
    free(h_kernel);
    free(h_out);

    dealloc_2d_matrix(img, num_rows, num_cols);
    dealloc_2d_matrix(kernel, kernel_size, kernel_size);
}