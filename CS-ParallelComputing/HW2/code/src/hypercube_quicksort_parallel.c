/*
 * @author      : Deniz Uzel
*/

#include <stdio.h>
#include <mpi.h>

#include <stdlib.h>
#include <stddef.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "custom_vector.h"

#define MASTER 0
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


void writeFile(char* output_file_name, int* A, int size){
    FILE * pFile;
    pFile = fopen(output_file_name, "wa");
    if (pFile == NULL){
        perror("Error opening file");
    }
    else{
        for(int i=0; i<size; i++){
                fprintf(pFile, "%d\n", A[i]);
            }
        }
    fclose(pFile);
}


void write_sorted_local_list_to_a_file(char* output_file_name, int* local_array, int local_array_size, int my_rank){
    char output_file_per_proc[128];
    snprintf(output_file_per_proc, sizeof(output_file_per_proc), "%s_part_%d", output_file_name, my_rank);
    writeFile(output_file_per_proc, local_array, local_array_size);
}


void plotArray(ARRAY_1D* A){
    int ending_index = A->size;

    for(int i=0; i<ending_index; i++){
        printf("\na[%d] = %d", i, A->arr[i]);
    }
    printf("\n");
}


int _partition(int* A, int p, int r, int pivot_index){
    int pivot_val = A[pivot_index];

    while (1){
        while (A[p] < pivot_val){
            p++;
        }

        while (pivot_val < A[r]){
            r--;
        }

        if (p <= r){
            // swap two values
            int tmp = A[r];
            A[r] = A[p];
            A[p] = tmp;

            p++;
            r--;
        } else{
            return p;
        }
    }
}


void _quick_sort(int* A, int p, int r){
    if (p < r){
        int pivot_index = p + (r-p)/2;
        int q = _partition(A, p, r, pivot_index);
        _quick_sort(A, p, q-1);
        _quick_sort(A, q, r);
    }

}


void QuickSort(int* A, int length){
    int last_index = length - 1;
    _quick_sort(A, 0, last_index);
}


int getMedian(int* A, int local_array_size){
    //printf("Median: %d\n", A[local_array_size/2-1]);
    return A[(local_array_size-1)/2];
}


void plot_arr(int* A, int size){
    printf("plot_arr() ->\n");
    for(int i=0; i<size; i++){
        printf("%d ", A[i]);
    }
    printf("\n");
}


int binarySearch(int* arr, int size, int target){
    /* adds 1 to target to find the cutting point for the upper array*/
    int low = 0;
    int high = size-1;
    target += 1;

    while (low <= high){
        int mid = low + (high-low)/2;

        if (arr[mid] > target)
            high = mid - 1;
        else if (arr[mid] < target)
            low = mid + 1;
        else
            return mid;
    }
    // if cutting position greater than 0 when array size is 1
    // lower array size will be 0
    return low;
}


void merge(int* merge_array, int* A, int a_size, int* B, int b_size){
    // first array should be big enought to contain A and B arrays
    int k = 0;
    int m = 0;

    int c = 0;
    while ((k<a_size) && (m<b_size)){
        if (A[k] < B[m]){
            merge_array[c] = A[k];
            k++;
        } else {
            merge_array[c] = B[m];
            m++;
        }
        c++;
    }

    while (k < a_size){
        merge_array[c] = A[k];
        k++;
        c++;
    }

    while (m < b_size){
        merge_array[c] = B[m];
        m++;
        c++;
    }
}


void valueCopy(int* output_array, int* output_array_size, int* array_to_copy, int array_to_copy_size){

    for(int i=0; i<array_to_copy_size; i++){
           output_array[i] = array_to_copy[i];
        }
    // update the output array size
    *output_array_size = array_to_copy_size;
}


void LastMerge(int* output_array, int* output_array_size, int* received_array, int received_size){
    int tmp_array_size = *output_array_size + received_size;
    int* tmp_array = (int*)malloc(sizeof(DATATYPE) * tmp_array_size);

    merge(tmp_array, output_array, *output_array_size, received_array, received_size);

    valueCopy(output_array, output_array_size, tmp_array, tmp_array_size);

    free(tmp_array);
}

void runParallel(int my_rank, int np, char* argv[])
{

    char* input_file_name = argv[1];
    char* output_file_name = argv[2];

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Status status;

    // input array
    ARRAY_1D* input_array;
    int input_array_size = -1;

    // dummy input array for array initialization for not getting a segmentation error
    if (my_rank != MASTER){
        input_array = Create1DArray(1);
    }

    // master process populates the input array
    if (my_rank == MASTER){
        // after exiting if statement, A becomes unreachable
        input_array = Create1DArray(CAPACITY);

        assert(input_array != NULL);

        readNums(input_file_name, input_array);
        input_array_size = input_array->size;
    }

    int* local_array;
    int local_array_size;

    // only the master sends the array size to all nodes
    MPI_Bcast(&input_array_size, 1, MPI_INT, MASTER, comm);

    // initialize local arrays
    local_array = (int*)malloc(sizeof(DATATYPE) * input_array_size);

    // Distributing
    int partition_size = input_array_size / np;
    local_array_size = partition_size;

    // chunks are automatically distributed by the scatter.
    MPI_Scatter(input_array->arr, partition_size, MPI_INT,
                local_array, partition_size, MPI_INT,
                MASTER, comm);

    QuickSort(local_array, local_array_size);

    // RECURSIVE PART
    int steps = log(np)/log(2);

    MPI_Comm new_comm;
    int comm_size;
    int color = 0;
    MPI_Comm_split(comm, color, my_rank, &new_comm);
    comm = new_comm;

    for(int i=0; i<steps; i++){
        MPI_Comm_size(comm, &comm_size);

        int number_of_medians = np / (i+1);
        int* medians_array = (int*)malloc(sizeof(DATATYPE) * number_of_medians);

        int median = getMedian(local_array, local_array_size);

        MPI_Allgather(&median, 1, MPI_INT,
                      medians_array, 1, MPI_INT,
                      comm);

        // sort medians
        QuickSort(medians_array, number_of_medians);

        // calculate partition value
        //plot_medians_array_horizontally(medians_array, number_of_medians, my_rank);
        int partition_value = getMedian(medians_array, number_of_medians);
        //printf("*rank: %d, median: %d\n", my_rank, partition_value);

        // find the partition indices
        int insert_point = binarySearch(local_array, local_array_size, partition_value);
        int lower_index;
        int upper_index;

        lower_index = insert_point -1;
        upper_index = insert_point;

        // determining partners for each iteration for the given number of steps
        int j = steps - i - 1;
        int partner = my_rank ^ (int)pow(2, j);

        int size_lower_send = lower_index + 1;
        int size_lower_received;

        int size_upper_send = local_array_size - upper_index;
        int size_upper_received;

        int* tmp_array;
        int tmp_array_size;

        // exchange lower arrays without deadlock
        // smaller my_rank id sends the upper array
        // greater my_rank id sends the lower array
        if (partner < my_rank){
            // send lower array size
            MPI_Send(&size_lower_send, 1, MPI_INT, partner, my_rank, comm);
            MPI_Recv(&size_upper_received, 1, MPI_INT, partner, partner, comm, &status);

            tmp_array_size = size_upper_received;
            tmp_array = (int*)malloc(sizeof(DATATYPE) * (tmp_array_size));

            // send the lower array
            MPI_Send(local_array, size_lower_send, MPI_INT, partner, my_rank, comm);
            MPI_Recv(tmp_array, size_upper_received, MPI_INT, partner, partner, comm, &status);

            int local_upper_size = local_array_size - upper_index;
            int* tmp_upper_part = (int*)malloc(sizeof(DATATYPE) * local_upper_size);

            for (int i=0; i<local_upper_size; i++){
                tmp_upper_part[i] = local_array[upper_index+i];
            }
            // merge with tmp array
            // local array from upper_index to local_array_size-1
            merge(local_array, tmp_upper_part, local_upper_size, tmp_array, tmp_array_size);

            // update the new array_size
            local_array_size = size_upper_received + local_upper_size;

            free(tmp_upper_part);
            free(tmp_array);

        } else{
            MPI_Recv(&size_lower_received, 1, MPI_INT, partner, partner, comm, &status);
            MPI_Send(&size_upper_send, 1, MPI_INT, partner, my_rank, comm);

            tmp_array_size = size_lower_received;
            tmp_array = (int*)malloc(sizeof(DATATYPE) * (tmp_array_size));

            MPI_Recv(tmp_array, size_lower_received, MPI_INT, partner, partner, comm, &status);
            MPI_Send(&(local_array[upper_index]), size_upper_send, MPI_INT, partner, my_rank, comm);

            // merge with tmp array
            // local array from 0 to lower_index+1

            int local_lower_size = lower_index + 1;
            int* tmp_lower_part = (int*)malloc(sizeof(DATATYPE) * local_lower_size);

            for (int i=0; i<local_lower_size; i++){
                tmp_lower_part[i] = local_array[i];
            }
            // merge with tmp array
            // local array from upper_index to local_array_size-1
            merge(local_array, tmp_lower_part, local_lower_size, tmp_array, tmp_array_size);

            // update the new array_size
            local_array_size = size_lower_received + local_lower_size;

            free(tmp_lower_part);
            free(tmp_array);
        }

        MPI_Comm_size(comm, &comm_size);

        if (my_rank < comm_size/2){
            color = 0;
        } else{
            color = 1;
        }

        int key;
        if (comm_size <= my_rank){
            key = my_rank - comm_size;
        } else {
            key = my_rank;
        }

        MPI_Comm_split(comm, color, key, &new_comm);

        comm = new_comm;
        MPI_Comm_rank(comm, &my_rank);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    comm = MPI_COMM_WORLD;

    write_sorted_local_list_to_a_file(output_file_name, local_array, local_array_size, my_rank);


    if (my_rank == MASTER){

        int output_array_capacity = input_array_size;
        int* output_array = (int*)malloc(sizeof(DATATYPE) * output_array_capacity);
        int output_array_size = 0;

        // copy the values of sorted array to output array from the
        // master's sorted array.

        valueCopy(output_array, &output_array_size, local_array, local_array_size);

        // copy the values of sorted array to output array from the
        // sorted arrays of non-master processors.
        int* received_array;
        int received_size;

        for(int p=1; p<np; p++){

            MPI_Recv(&received_size, 1, MPI_INT,
                     p, p, comm, &status);

            received_array = (int*)malloc(sizeof(DATATYPE) * received_size);

            MPI_Recv(received_array, received_size, MPI_INT,
                     p, p, comm, &status);

            LastMerge(output_array, &output_array_size, received_array, received_size);

        }
        free(received_array);

        // write final sorted output array to file with given output file argument
        writeFile(output_file_name, output_array, output_array_size);

    } else {

        MPI_Send(&local_array_size, 1, MPI_INT,
                 MASTER, my_rank, comm);

        MPI_Send(local_array, local_array_size, MPI_INT,
                 MASTER, my_rank, comm);
    }

    free(local_array);

    free_1DArray(input_array);
}

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int np;
    int my_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    runParallel(my_rank, np, argv);

    MPI_Finalize();
}
