
cimport numpy as np

ctypedef np.float_t FLOAT_t
# Cython does not allow this:
# ctypedef np.ndarray[FLOAT_t, ndim=1] VECTOR_t
# ctypedef np.ndarray[FLOAT_t, ndim=2] ARRAY_t

# FIXHIM: no overloading in Cython
cdef np.ndarray[FLOAT_t, ndim=1] softmax(np.ndarray[FLOAT_t, ndim=1] a, np.ndarray out=*)
cdef np.ndarray[FLOAT_t, ndim=2] softmax2d(np.ndarray[FLOAT_t, ndim=2] a, int axis=*, np.ndarray out=*)

# FIXHIM: no overloading in Cython
cdef FLOAT_t logsumexp(np.ndarray[FLOAT_t] a)
cdef np.ndarray[FLOAT_t, ndim=1] logsumexp2d(np.ndarray[FLOAT_t, ndim=2] a, int axis=*)

cdef np.ndarray[FLOAT_t, ndim=1] tanh(np.ndarray[FLOAT_t, ndim=1] weights,
                                      np.ndarray out=*)
cdef np.ndarray[FLOAT_t, ndim=1] tanhe(np.ndarray[FLOAT_t, ndim=1] y,
                                       np.ndarray out=*)

cdef np.ndarray[FLOAT_t, ndim=1] hardtanh(np.ndarray[FLOAT_t, ndim=1] weights,
                                          np.ndarray out=*)

cdef np.ndarray[FLOAT_t, ndim=1] hardtanhd(np.ndarray[FLOAT_t, ndim=1] y,
                                   np.ndarray out=*)
cdef np.ndarray[FLOAT_t, ndim=2] hardtanhd2d(np.ndarray[FLOAT_t, ndim=2] y,
                                   np.ndarray out=*)

cdef np.ndarray[FLOAT_t, ndim=1] hardtanh_back(np.ndarray[FLOAT_t, ndim=1] y,
                                           np.ndarray[FLOAT_t, ndim=1] grads,
                                           np.ndarray[FLOAT_t, ndim=1] grads_in)

cdef np.ndarray[FLOAT_t, ndim=2] hardtanh_back2d(np.ndarray[FLOAT_t, ndim=2] y,
                                                 np.ndarray[FLOAT_t, ndim=2] grads,
                                                 np.ndarray[FLOAT_t, ndim=2] grads_in)

cdef np.ndarray[FLOAT_t, ndim=1] hardtanhe(np.ndarray[FLOAT_t, ndim=1] y,
                                   np.ndarray out=*)
cdef np.ndarray[FLOAT_t, ndim=2] hardtanhe2d(np.ndarray[FLOAT_t, ndim=2] y,
                                   np.ndarray out=*)

