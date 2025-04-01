/*
@author: Deniz Uzel
*/

#include <stdio.h>
#include <stdlib.h>
#include "custom_vector.h"


ARRAY_1D* Create1DArray(int capacity){
    ARRAY_1D* A = (ARRAY_1D*)malloc(sizeof(ARRAY_1D));

    if (A == NULL){
        return NULL;
    }

    A->arr = (DATATYPE*)malloc(capacity * sizeof(DATATYPE));
    if (A->arr == NULL) {
        free(A);
        return NULL;
    }

    A->capacity = capacity;
    A->size = 0;

    return A;
}


void free_1DArray(ARRAY_1D* A){
    if (A != NULL){
        // defensive control
        if (A->arr != NULL){
            free(A->arr);
        }
        free(A);
    } else {
        printf("free_1DArray error.");
    }
}


ARRAY_OF_PAIRS* CreateArrayOfPairs(int pairCapacity){
    if (pairCapacity % 2 != 0) {
        printf("Capacity must be even.\n");
        return NULL;
    }

    ARRAY_OF_PAIRS* A = (ARRAY_OF_PAIRS*)malloc(sizeof(ARRAY_OF_PAIRS));

    if (A == NULL){
        return NULL;
    }

    A->arr = (DATATYPE**)malloc(pairCapacity * sizeof(DATATYPE*));
    if (A->arr == NULL) {
        free(A);
        return NULL;
    }

    for (int i = 0; i < pairCapacity; i++){
        A->arr[i] = (DATATYPE*)malloc(2 * sizeof(DATATYPE));
        if (A->arr[i] == NULL) {
            // free the allocated part if it fails
            for (int j = 0; j < i; j++){
                free(A->arr[j]);
            }
            free(A->arr);
            free(A);
            return NULL;
        }
    }

    A->pairCapacity = pairCapacity;
    A->pairSize = 0;

    return A;
}


void free_ArrayOfPairs(ARRAY_OF_PAIRS* A){
    if (A != NULL){
        for (int i = 0; i < A->pairCapacity; i++){
            free(A->arr[i]);
        }

        // defensive control
        if (A->arr != NULL){
            free(A->arr);
        }

        free(A);
    } else {
        printf("free_ArrayOfPairs error.");
    }
}
