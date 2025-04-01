/*
 * @author      : Deniz Uzel
*/

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include <custom_vector.h>


#define BUFFER_SIZE 100
#define CAPACITY 1024


void readNums(char* file_name, ARRAY_1D* A){
    FILE * pFile;
    char mystring [BUFFER_SIZE];
    int num_read = 0;

    // Read a file
    pFile = fopen(file_name, "r");
    if (pFile == NULL){
        perror("Error opening file");
    }
    else{
        int i = 0;
        while ((fgets(mystring, BUFFER_SIZE, pFile) != NULL )){
            // assuming that capacity of the array is enough for the input array
            num_read = atoi(mystring);
            A->arr[i] = num_read;
            i += 1;
        A->size = i;
        }
    }
    fclose(pFile);
}


void writeFile(char* output_file_name, ARRAY_1D* A){
    FILE * pFile;
    pFile = fopen(output_file_name, "wa");
    if (pFile == NULL){
        perror("Error opening file");
    }
    else{
        for(int i=0; i<A->size; i++){
                fprintf(pFile, "%d\n", A->arr[i]);
            }
        }
    fclose(pFile);
}


int _partition(ARRAY_1D* A, int p, int r, int pivot_index){
    int pivot_val = A->arr[pivot_index];

    while (1){
        while (A->arr[p] < pivot_val){
            p++;
        }

        while (pivot_val < A->arr[r]){
            r--;
        }

        if (p <= r){
            // swap two values
            int tmp = A->arr[r];
            A->arr[r] = A->arr[p];
            A->arr[p] = tmp;

            p++;
            r--;
        } else{
            return p;
        }
    }
}


void _quick_sort(ARRAY_1D* A, int p, int r){
    if (p < r){
        int pivot_index = p + (r-p)/2;
        int q = _partition(A, p, r, pivot_index);
        _quick_sort(A, p, q-1);
        _quick_sort(A, q, r);
    }

}


void QuickSort(ARRAY_1D* A){
    int last_index = A->size - 1;
    _quick_sort(A, 0, last_index);
}


int main(int argc, char *argv[]){
    ARRAY_1D* A = Create1DArray(CAPACITY);
    assert(A != NULL);

    char* file_name = argv[1];
    char* output_file_name = argv[2];

    // read numbers from input.txt and populates the dynamic array A
    readNums(file_name, A);

    // sort the array
    QuickSort(A);

    // write the array into output file
    writeFile(output_file_name, A);
}
