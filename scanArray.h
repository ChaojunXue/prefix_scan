
#ifndef scanArray_h
#define scanArray_h

void exclusive_scan(size_t *outArray, const size_t *inArray, size_t numElements, size_t *d_lastSumVal);
void exclusive_scan(int *outArray, const int *inArray, size_t numElements, int *d_lastSumVal);

#endif /* scanArray_h */
