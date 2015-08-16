
cimport numpy as np

ctypedef np.float_t FLOAT_t

# FIXHIM: no overloading in Cython
cdef float[:] softmax(float[:] a, float[:] out=*)
cdef float[:,:] softmax2d(float[:,:] a, int axis=*, float[:,:] out=*)

# FIXHIM: no overloading in Cython
cdef logsumexp(np.ndarray[FLOAT_t] a)
cdef logsumexp2d(np.ndarray[FLOAT_t, ndim=2] a, int axis=*)

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

cdef np.ndarray[FLOAT_t, ndim=1] hardtanhe(np.ndarray[FLOAT_t, ndim=1] y,
                                   np.ndarray out=*)
cdef np.ndarray[FLOAT_t, ndim=2] hardtanhe2d(np.ndarray[FLOAT_t, ndim=2] y,
                                   np.ndarray out=*)

