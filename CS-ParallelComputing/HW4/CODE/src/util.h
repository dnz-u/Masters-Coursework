//
// Created by utku on 19.11.2022.
//

#include <cstdlib>
#include <cstring>
#include <cstdio>



/*
*	Allocates an rxc integer matrix
*/
int** alloc_2d_matrix(int r, int c);

/*
*	Deallocates an rxc integer matrix
*/
void dealloc_2d_matrix(int ** a, int r, int c);

/*
* Prints an rxc integer matrix
*/
void print_2d_matrix(int **a, int r, int c);

/*@params:
*		file_name: name of the file
*		h: number of rows in the file
*		w: number of columns in the file
*		reads a matrix file
*		note that this function can not read pgm files, only use with given dataset
*       Usage Example:
*       int num_rows,num_columns
*       int **img;
*       char* file_name = argv[1];
*       img = read_pgm_file(argv[1],&num_rows,&num_columns);
**/

void print_2d_matrix_to_file(int **a, int r, int c, FILE *f);

void print_1d_matrix_to_file(int **a, int r, int c, FILE *f);

int** read_pgm_file(char * file_name, int *num_rows, int *num_columns);



// ADDITIONAL FUNCTIONS IMPORTED from PROJECT 3

int* extend_edges(int **img, int row_index, int col_index, int extend_amount);


void writeFile(char* output_file_name, int** arr, int r, int c);
void writeFile_1d(char* output_file_name, int* arr, int r, int c);


// Controls CUDA output with project3 sequential implementation
void checkResults(char* file_name, int* comparison_img);


int* alloc_1d_matrix(int r, int c);

void print_1d_matrix(int *a, int r, int c);

int* convert_kernel_2d_to_1d(int** a, int r, int c);

