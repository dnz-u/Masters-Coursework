/*
@author: Deniz Uzel
*/

#include <cstdio>
#include <omp.h>
#include "util.h"
#include <time.h>
#include <cmath>


int around(double);
void writeFile(char*, int**, int, int);


void set_zero(int **a, int r, int c);

int** extend_edges(int **img, int row_index, int col_index, int extend_amount){
    int num_rows = row_index + 1;
    int num_cols = col_index + 1;

    int new_num_rows = num_rows + extend_amount*2;
    int new_num_cols = num_cols + extend_amount*2;


    int** extended_img = alloc_2d_matrix(new_num_rows, new_num_cols);

    // parellelized
    set_zero(extended_img, new_num_rows, new_num_cols);

    // copy pixel val of the source img into extended img
    #pragma omp parallel for
    for (int i=0; i<num_rows; i++){
        for (int j=0; j<num_cols; j++){
            extended_img[i+extend_amount][j+extend_amount] = img[i][j];
        }
    }

    // extend the upper and lower rows
    #pragma omp parallel for
    for (int i=0; i<extend_amount; i++){
        for (int j=0; j<num_cols; j++){
            // extends upper row
            extended_img[i][j+extend_amount] = img[i][j];
            // extends lower rows
            extended_img[new_num_rows-1-i][j+extend_amount] = img[num_rows-1-i][j];
        }
    }

    // extend the left and right columns
    #pragma omp parallel for
    for (int j=0; j<extend_amount; j++){
        for (int i=0; i<num_rows; i++) {
            // extends right columns
            extended_img[i+extend_amount][j] = img[i][j];
            // extends right columns
            extended_img[i+extend_amount][new_num_cols-1-j] = img[i][num_cols-1-j];

        }
    }

    // add corners, upper left, lower right
    #pragma omp parallel for
    for (int i=0; i<extend_amount; i++){
        extended_img[i][i] = img[i][i];
        extended_img[new_num_rows-1-i][new_num_cols-1-i] = img[num_rows-1-i][num_cols-1-i];
    }

    // add corners, lower left
    #pragma omp parallel for
    for (int i=new_num_rows-1; new_num_rows-1-extend_amount<i; i--){
        for (int j=0; j<extend_amount; j++){
            extended_img[i][j] = img[i-extend_amount*2][j];
        }
    }

    // add corners, upper right
    #pragma omp parallel for
    for (int j=new_num_cols-1; new_num_cols-1-extend_amount<j; j--){
        for (int i=0; i<extend_amount; i++){
            extended_img[i][j] = img[i][j-extend_amount*2];
        }
    }


    return extended_img;
}


int average_operation(int kernel_size, int **img, int row_index, int col_index){
    int sum_val = 0;

    for (int i=row_index; i < (row_index+kernel_size); i++){
        for (int j=col_index; j < (col_index+kernel_size); j++){
            sum_val += img[i][j];
        }
    }
    double avg_val = (double)sum_val / (kernel_size*kernel_size);

    return avg_val;
}


int **denoise_image(int kernel_size, int **img, int num_rows, int num_cols){
    // EXTENTION
    int** extended_img;
    int extend_amount = kernel_size / 2;

    extended_img = extend_edges(img, num_rows-1, num_cols-1, extend_amount);

    int extended_num_rows = num_rows;
    int extended_num_columns = num_cols;

    extended_num_rows += extend_amount*2;
    extended_num_columns += extend_amount*2;

    // new_img
    int** new_img = alloc_2d_matrix(num_rows, num_cols);

    // perform averaging
    #pragma omp parallel
    {
        #pragma omp for
        for (int i=0; i<num_rows; i++){
            for (int j=0; j<num_cols; j++){
                int val = average_operation(kernel_size, extended_img, i, j);
                new_img[i][j] = val;
            }
        }
    }

    dealloc_2d_matrix(extended_img, extended_num_rows, extended_num_columns);

    return new_img;
}


int motion_blur(int **kernel, int kernel_size, int **img, int row_index, int col_index){
    int sum_val = 0;
    for (int i=row_index; i < (row_index+kernel_size); i++){
        for (int j=col_index; j < (col_index+kernel_size); j++){
            sum_val = sum_val + img[i][j] * kernel[i-row_index][j-col_index];
        }
    }

    int kernel_sum = 0;
    for (int i=0; i < kernel_size; i++){
        for (int j=0; j < kernel_size; j++){
            kernel_sum += kernel[i][j];
        }
    }

    double normalized_val = (double)sum_val / kernel_sum;
    return normalized_val;
}


int **blur_image(int **kernel, int kernel_size, int **img, int num_rows, int num_cols){
    // EXTENTION
    int** extended_img;
    int extend_amount = kernel_size / 2;

    extended_img = extend_edges(img, num_rows-1, num_cols-1, extend_amount);

    int extended_num_rows = num_rows;
    int extended_num_columns = num_cols;

    extended_num_rows += extend_amount*2;
    extended_num_columns += extend_amount*2;

    // new_img
    int** new_img = alloc_2d_matrix(num_rows, num_cols);

    int i,j;
    // perform convolutions
    #pragma omp parallel private(i,j)
    {
        #pragma omp for
        for (i=0; i<num_rows; i++){
            for (j=0; j<num_cols; j++){
                int val = motion_blur(kernel, kernel_size, extended_img, i, j);
                new_img[i][j] = val;
            }
        }
    }

    dealloc_2d_matrix(extended_img, extended_num_rows, extended_num_columns);

    return new_img;
}


int main(int argc, char* argv[]) {

    int num_rows, num_columns, kernel_size;

    int **matrix = read_pgm_file(argv[1], &num_rows, &num_columns);

    int **kernel = read_pgm_file(argv[2], &kernel_size, &kernel_size);

    char* file_name_denoised = argv[3];
    char* file_name_blurred = argv[4];

    int **denoised_image_omp;
    int **blurred_image_omp;


    // PARALLELIZED CALLS
    // time
    double t;

    t = omp_get_wtime();
    denoised_image_omp = denoise_image(kernel_size, matrix, num_rows, num_columns);
    double time_parallel_denoise = (omp_get_wtime() - t) * 1000; // in seconds

    t = omp_get_wtime();
    blurred_image_omp = blur_image(kernel, kernel_size, matrix, num_rows, num_columns);
    double time_parallel_blur = (omp_get_wtime() - t) * 1000; // in seconds


    // PRINT RUNNING TIME
    printf("\n");
    printf("Parallel time, denoise: %.2f ms\n", time_parallel_denoise);
    printf("Parallel time, blur: %.2f ms\n", time_parallel_blur);
    printf("\n");


    // PRINT MATRICES
    //print_2d_matrix(denoised_image_omp, num_rows, num_columns);
    //print_2d_matrix(blurred_image_omp, num_rows, num_columns);


    // WRITE TO FILE
    // write denoised image to file
    //writeFile(file_name_denoised, denoised_image_omp, num_rows, num_columns);
    // write blurred image to file
    //writeFile(file_name_blurred, blurred_image_omp, num_rows, num_columns);


    // DEALLOC MEMORY
    dealloc_2d_matrix(denoised_image_omp, num_rows, num_columns);
    dealloc_2d_matrix(blurred_image_omp, num_rows, num_columns);

    dealloc_2d_matrix(matrix, num_rows, num_columns);
    dealloc_2d_matrix(kernel, kernel_size, kernel_size);
     
    return 0;
}

void set_zero(int **a, int r, int c){
    #pragma omp parallel for
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c;j++ ){
            a[i][j] = 0;
        }
    }
}


// HELPER FUNCTIONS
int around(double x){
    double dec = x - (int)x;

    if (dec < 0.5)
        return (int)x;
    else
        return (int)x + 1;
}


void writeFile(char* output_file_name, int** arr, int r, int c){
    FILE * pFile;
    pFile = fopen(output_file_name, "w");
    if (pFile == NULL){
        perror("Error opening file");
    }
    else{
        print_2d_matrix_to_file(arr, r, c, pFile);
        }
    fclose(pFile);
}
