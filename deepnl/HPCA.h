
#ifndef deepnl_HPCA_H
#define deepnl_HPCA_H

#include "Python.h"

namespace hpca {

bool matrix_sqrt(float* m, int rowstart, int rowend, int cols);

bool distance_matrix(float* m, int rowstart, int rowend, int rows, int cols,
		     float* dm);

PyObject* cooccurrence_matrix(char* corpus, char* vocabFile, unsigned top,
			      unsigned window);

void hellinger_matrix(float* dm, float* cooccur, int rows, int cols, int lines);

extern bool verbose;
}

#endif // deepnl_HPCA_H
