# distutils: language = c++

cimport numpy as np
import numpy as np

# not supported by Cython:
# ctypedef np.ndarray[FLOAT_t, ndim=1] np.ndarray[FLOAT_t,ndim=1]
# ctypedef np.ndarray[FLOAT_t, ndim=2] np.ndarray[FLOAT_t,ndim=2]

# ----------------------------------------------------------------------
# Math functions

cdef softmax(np.ndarray[FLOAT_t,ndim=1] a):
    """Compute the ratio of exp(a) to the sum of exponentials.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    res : ndarray
        The result, ``np.exp(a)/(np.sum(np.exp(a), axis))`` calculated in a numerically stable way.
    """
    a_max = a.max()
    e = np.exp(a - a_max)
    return e / np.sum(e)

cdef softmax2d(np.ndarray[FLOAT_t,ndim=2] a, int axis=0):
    """Compute the ratio of exp(a) to the sum of exponentials along the axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis over which the sum is taken. By default `axis` is 0,

    Returns
    -------
    res : ndarray
        The result, ``np.exp(a)/(np.sum(np.exp(a), axis))`` calculated in a numerically stable way.
    """
    a_max = a.max(axis)
    e = np.exp(a - a_max)
    return e / np.sum(e, axis)

cdef logsumexp(np.ndarray[FLOAT_t,ndim=1] a):
    """Compute the log of the sum of exponentials of input elements.
    like: scipy.misc.logsumexp

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way.
    """
    a_max = a.max()
    return np.log(np.sum(np.exp(a - a_max))) + a_max

cdef logsumexp2d(np.ndarray[FLOAT_t,ndim=2] a, axis=None):
    """Compute the log of the sum of exponentials of input elements.
    like: scipy.misc.logsumexp

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis over which the sum is taken. By default `axis` is None,
        and all elements are summed.

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way.
    """
    if axis is None:
        a = a.ravel()
    else:
        a = np.rollaxis(a, axis)
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max

cdef np.ndarray[FLOAT_t, ndim=1] hardtanh(np.ndarray[FLOAT_t, ndim=1] weights,
                                          np.ndarray out = None):
    """Hard hyperbolic tangent."""
    if out is None:
        out = np.empty_like(weights)
    cdef int i
    cdef double w
    for i, w in enumerate(weights):
        if w < -1:
            out[i] = -1
        elif w > 1:
            out[i] = 1
        else:
            out[i] = w
    return out

cdef np.ndarray[FLOAT_t, ndim=1] hardtanhd(np.ndarray[FLOAT_t, ndim=1] weights,
                                   np.ndarray out=None):
    """derivative of hardtanh"""
    if out is None:
        out = np.empty_like(weights)
    cdef int i
    cdef double w
    for i, w in enumerate(weights.flat):
        if -1.0 <= w <= 1.0:
            out.flat[i] = 1.0
        else:
            out.flat[i] = 0.0
    return out

cdef np.ndarray[FLOAT_t, ndim=2] hardtanhd2d(np.ndarray[FLOAT_t, ndim=2] weights,
                                   np.ndarray out=None):
    """derivative of hardtanh"""
    if out is None:
        out = np.empty_like(weights)
    cdef int i
    cdef double w
    for i, w in enumerate(weights.flat):
        if -1.0 <= w <= 1.0:
            out.flat[i] = 1.0
        else:
            out.flat[i] = 0.0
    return out

cdef np.ndarray[FLOAT_t, ndim=1] hardtanhe(np.ndarray[FLOAT_t, ndim=1] y,
                                   np.ndarray out=None):
    """derivative of hardtanh in terms of y = hardtanh(x) ="""
    if out is None:
        out = np.ones_like(y)
    cdef int i
    cdef double w
    for i, w in enumerate(y.flat):
        if  w == -1.0 or w == 1.0:
            out.flat[i] = 0.0
        else:
            out.flat[i] = 1.0
    return out

cdef np.ndarray[FLOAT_t, ndim=2] hardtanhe2d(np.ndarray[FLOAT_t, ndim=2] y,
                                   np.ndarray out=None):
    """derivative of hardtanh in terms of y = hardtanh(x) ="""
    if out is None:
        out = np.ones_like(y)
    cdef int i
    cdef double w
    for i, w in enumerate(y.flat):
        if  w == -1.0 or w == 1.0:
            out.flat[i] = 0.0
        else:
            out.flat[i] = 1.0
    return out
