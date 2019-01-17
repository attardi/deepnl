# distutils: language=c++

cimport numpy as np

from network cimport float_t, int_t

# FIXHIM: no overloading in Cython
cdef np.ndarray[float_t] softmax(np.ndarray[float_t] a, np.ndarray out=*)
cdef np.ndarray[float_t, ndim=2] softmax2d(np.ndarray[float_t, ndim=2] a, int_t axis=*, np.ndarray out=*)

# FIXHIM: no overloading in Cython
cdef float_t logsumexp(np.ndarray[float_t] a)
cdef np.ndarray[float_t] logsumexp2d(np.ndarray[float_t, ndim=2] a, int_t axis=*)

cdef np.ndarray[float_t] tanh(np.ndarray[float_t] weights,
                                      np.ndarray out=*)
cdef np.ndarray[float_t] tanhe(np.ndarray[float_t] y,
                                       np.ndarray out=*)

cdef np.ndarray[float_t] hardtanh(np.ndarray[float_t] weights,
                                          np.ndarray out=*)

cdef np.ndarray[float_t] hardtanhd(np.ndarray[float_t] y,
                                   np.ndarray out=*)
cdef np.ndarray[float_t, ndim=2] hardtanhd2d(np.ndarray[float_t, ndim=2] y,
                                   np.ndarray out=*)

cdef np.ndarray[float_t] hardtanh_back(np.ndarray[float_t] y,
                                           np.ndarray[float_t] grads,
                                           np.ndarray[float_t] grads_in)

cdef np.ndarray[float_t, ndim=2] hardtanh_back2d(np.ndarray[float_t, ndim=2] y,
                                                 np.ndarray[float_t, ndim=2] grads,
                                                 np.ndarray[float_t, ndim=2] grads_in)

cdef np.ndarray[float_t] hardtanhe(np.ndarray[float_t] y,
                                   np.ndarray out=*)
cdef np.ndarray[float_t, ndim=2] hardtanhe2d(np.ndarray[float_t, ndim=2] y,
                                   np.ndarray out=*)

