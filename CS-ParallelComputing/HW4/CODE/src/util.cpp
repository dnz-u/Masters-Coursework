#include "util.h"
#include <cassert>


int* alloc_1d_matrix(int r, int c)
{
    int *a;
    int i = 0;
    a = (int*)malloc(sizeof(int)*r*c);

    if (a == NULL){
        perror("memory allocation failure");
        exit(0);
    }

    return a;
}


/* converts 2d kernel into 1d dynamic array.
   small size of the kernel makes the operation negligible.
*/
int* convert_kernel_2d_to_1d(int** a, int r, int c){
    int* arr_1d = alloc_1d_matrix(r, c);
    for (int i = 0; i < r; i++){
        for (int j = 0; j < c; j++){
            arr_1d[i*c + j] = a[i][j];
            }
    }
    return arr_1d;
}


void print_1d_matrix(int *a, int r, int c){
    for (int i=0; i < r*c; i++){
        if (i%c == 0)
            printf("\n");
        printf("%d ", a[i]);
    }
    printf("\n");
}


int **alloc_2d_matrix(int r, int c)
{
    int ** a;
    int i;
    a = (int **)malloc(sizeof(int *) * r);
    if (a == NULL) {
        perror("memory allocation failure");
        exit(0);
    }
    for (i = 0; i < r; ++i) {
        a[i] = (int *)malloc(sizeof(int) * c);
        if (a[i] == NULL) {
            perror("memory allocation failure");
            exit(EXIT_FAILURE);
        }
    }
    return a;
}


void dealloc_2d_matrix(int **a, int r, int c)
{
    int i;
    for (i = 0; i < r; ++i)
        free(a[i]);
    free(a);
}


void print_2d_matrix_to_file(int **a, int r, int c, FILE *f){
    fprintf(f,"%d\n",r );
    fprintf(f,"%d\n",c );
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c;j++ ){
            fprintf(f,"%d ",a[i][j] );
        }
        fprintf(f,"\n");
    }
}


void print_1d_matrix_to_file(int *a, int r, int c,  FILE *f){
    fprintf(f,"%d\n",r );
    fprintf(f,"%d",c );
    for (int i=0; i < r*c; i++){
        if (i%c == 0)
            fprintf(f,"\n");
        fprintf(f, "%d ", a[i]);
    }
}


int** read_pgm_file(char * file_name, int *num_rows, int *num_columns)
{
    FILE *inputFile = fopen(file_name, "r");
    if(inputFile){
        int success;
        success = fscanf(inputFile, "%d", &*num_rows);
        if(!success){
            printf("Bad File format!\n");
            return(NULL);
        }
        success = fscanf(inputFile, "%d", &*num_columns);
        if(!success){
            printf("Bad File format!\n");
            return(NULL);
        }

        int i,j, int_tmp;
        int** data=alloc_2d_matrix(*num_rows,*num_columns);

        for (i = 0; i < (*num_rows); i++)
        {
            for (j = 0; j < (*num_columns); j++) {
                fscanf(inputFile,"%d", &int_tmp);
                data[i][j] = int_tmp;
            }
        }
        fclose(inputFile);
        return data;
    }
    return(NULL);
}


// ADDITIONAL FUNCTIONS IMPORTED from PROJECT 3

int* extend_edges(int **img, int row_index, int col_index, int extend_amount){
    int num_rows = row_index + 1;
    int num_cols = col_index + 1;

    int new_num_rows = num_rows + extend_amount*2;
    int new_num_cols = num_cols + extend_amount*2;


    int* extended_img = alloc_1d_matrix(new_num_rows, new_num_cols);


    // copy pixel val of the source img into extended img
    for (int i=0; i<num_rows; i++){
        for (int j=0; j<num_cols; j++){
            extended_img[(i+extend_amount)*new_num_cols + j+extend_amount] = img[i][j];
        }
    }

    // extend the upper and lower rows
    for (int i=0; i<extend_amount; i++){
        for (int j=0; j<num_cols; j++){
            // extends upper row
            extended_img[i*new_num_cols + j+extend_amount] = img[i][j];
            // extends lower rows
            extended_img[(new_num_rows-1-i)*new_num_cols + j+extend_amount] = img[num_rows-1-i][j];
        }
    }

    // extend the left and right columns
    for (int j=0; j<extend_amount; j++){
        for (int i=0; i<num_rows; i++) {
            // extends right columns
            extended_img[(i+extend_amount)*new_num_cols + j] = img[i][j];
            // extends right columns
            extended_img[(i+extend_amount)*new_num_cols + new_num_cols-1-j] = img[i][num_cols-1-j];

        }
    }

    // add corners, upper left, lower right
    for (int i=0; i<extend_amount; i++){
        extended_img[i*new_num_cols + i] = img[i][i];
        extended_img[(new_num_rows-1-i)*new_num_cols + new_num_cols-1-i] = img[num_rows-1-i][num_cols-1-i];
    }

    // add corners, lower left
    for (int i=new_num_rows-1; new_num_rows-1-extend_amount<i; i--){
        for (int j=0; j<extend_amount; j++){
            extended_img[i*new_num_cols + j] = img[i-extend_amount*2][j];
        }
    }

    // add corners, upper right
    for (int j=new_num_cols-1; new_num_cols-1-extend_amount<j; j--){
        for (int i=0; i<extend_amount; i++){
            extended_img[i*new_num_cols + j] = img[i][j-extend_amount*2];
        }
    }

    return extended_img;
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


void writeFile_1d(char* output_file_name, int* arr, int r, int c){
    FILE * pFile;
    pFile = fopen(output_file_name, "w");
    if (pFile == NULL){
        perror("Error opening file");
    }
    else{
        print_1d_matrix_to_file(arr, r, c, pFile);
        }
    fclose(pFile);
}

void checkResults(char* file_name, int* comparison_img){
    int num_rows, num_cols;

    int** matrix = read_pgm_file(file_name, &num_rows, &num_cols);

    int correct_val = 0;
    int is_wrong = 0;

    // comparison
    for (int i=0; i<num_rows; i++){
        for (int j=0; j<num_cols; j++){
            if (matrix[i][j] == comparison_img[i*num_cols + j]){
                correct_val += 1;
            } else{
                is_wrong = 1;
            }
        }
    }

    printf("correct-labels: %d / %d \n", correct_val, num_rows*num_cols);

    if (is_wrong){
        assert(1 == 0);
    }
    printf("\n@.@> Result is CORRECT.\n\n");
}
