/*
@author: Deniz Uzel
It is used to read files of unknown size.
*/
#ifndef CUSTOM_VECTOR_H
#define CUSTOM_VECTOR_H


#define DATATYPE int


typedef struct {
    int capacity;
    int size;
    DATATYPE* arr;
} ARRAY_1D;
ARRAY_1D* Create1DArray(int capacity);
void free_1DArray(ARRAY_1D* array);


typedef struct {
    int pairCapacity;
    int pairSize;
    DATATYPE** arr;
} ARRAY_OF_PAIRS;
ARRAY_OF_PAIRS* CreateArrayOfPairs(int pairCapacity);
void free_ArrayOfPairs(ARRAY_OF_PAIRS* array);


#endif

