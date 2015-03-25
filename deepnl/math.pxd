
cimport numpy as np

ctypedef np.float_t FLOAT_t

# FIXHIM: no overloading in Cython
cdef softmax(np.ndarray[FLOAT_t] a)
cdef softmax2d(np.ndarray[FLOAT_t, ndim=2] a, int axis=*)

# FIXHIM: no overloading in Cython
cdef logsumexp(np.ndarray[FLOAT_t] a)
cdef logsumexp2d(np.ndarray[FLOAT_t, ndim=2] a, axis=*)

cdef np.ndarray[FLOAT_t, ndim=1] hardtanh(np.ndarray[FLOAT_t, ndim=1] weights,
                                          np.ndarray out=*)

cdef np.ndarray[FLOAT_t, ndim=1] hardtanhd(np.ndarray[FLOAT_t, ndim=1] y,
                                   np.ndarray out=*)
cdef np.ndarray[FLOAT_t, ndim=2] hardtanhd2d(np.ndarray[FLOAT_t, ndim=2] y,
                                   np.ndarray out=*)

cdef np.ndarray[FLOAT_t, ndim=1] hardtanhe(np.ndarray[FLOAT_t, ndim=1] y,
                                   np.ndarray out=*)
cdef np.ndarray[FLOAT_t, ndim=2] hardtanhe2d(np.ndarray[FLOAT_t, ndim=2] y,
                                   np.ndarray out=*)

