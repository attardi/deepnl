# distutils: language = c++

cimport numpy as np
import numpy as np

# ----------------------------------------------------------------------
# Math functions

cdef np.ndarray[FLOAT_t, ndim=1] softmax(np.ndarray[FLOAT_t, ndim=1] a, np.ndarray out=None):
    """Compute the ratio of exp(a) to the sum of exponentials.

    Parameters
    ----------
    a : array_like
        Input array.
    out : array_like, optional
        Alternative output array in which to place the result.

    Returns
    -------
    res : ndarray
        The result, ``np.exp(a)/(np.sum(np.exp(a), axis))`` calculated in a numerically stable way.
    """
    if out is None:
        out = np.empty_like(a)
    np.exp(a - a.max(), out)
    out /= np.sum(out)
    return out

cdef np.ndarray[FLOAT_t, ndim=2] softmax2d(np.ndarray[FLOAT_t, ndim=2] a, int axis=0, np.ndarray out=None):
    """Compute the ratio of exp(a) to the sum of exponentials along the axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis over which the sum is taken. By default `axis` is 0,
    out : array_like, optional
        Alternative output array in which to place the result.

    Returns
    -------
    res : ndarray
        The result, ``np.exp(a)/(np.sum(np.exp(a), axis))`` calculated in a numerically stable way.
    """
    if out is None:
        out = np.empty_like(a)
    np.exp(a - a.max(axis), out)
    out /= np.sum(out, axis)
    return out

cdef FLOAT_t logsumexp(np.ndarray[FLOAT_t,ndim=1] a):
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

cdef np.ndarray[FLOAT_t, ndim=1] logsumexp2d(np.ndarray[FLOAT_t,ndim=2] a, int axis=0):
    """Compute the log of the sum of exponentials of input elements.
    like: scipy.misc.logsumexp

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int
        Axis over which the sum is taken.

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way.
    """
    a = np.rollaxis(a, axis)
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max

cdef np.ndarray[FLOAT_t, ndim=1] tanh(np.ndarray[FLOAT_t, ndim=1] weights,
                                      np.ndarray out=None):
    """Hyperbolic tangent.
    out : array_like, optional
        Alternative output array in which to place the result.
    """
    if out is None:
        out = np.empty_like(weights)
    np.tanh(weights, out)
    return out

cdef np.ndarray[FLOAT_t, ndim=1] tanhe(np.ndarray[FLOAT_t, ndim=1] y,
                                       np.ndarray out=None):
    """Hyperbolic tangent.
    out : array_like, optional
        Alternative output array in which to place the result.
    """
    if out is None:
        out = np.empty_like(y)
    out[:] = 1 - y**2
    return out

cdef np.ndarray[FLOAT_t, ndim=1] hardtanh(np.ndarray[FLOAT_t, ndim=1] y,
                                          np.ndarray out=None):
    """Hard hyperbolic tangent.
    out : array_like, optional
        Alternative output array in which to place the result.
    """
    it = np.nditer([y, out],
                   op_flags = [['readonly'],
                               ['writeonly', 'allocate', 'no_broadcast']])
    for w, o in it:
        if w < -1:
            o[...] = -1
        elif w > 1:
            o[...] = 1
        else:
            o[...] = w
    return it.operands[1]

cdef np.ndarray[FLOAT_t, ndim=1] hardtanhd(np.ndarray[FLOAT_t, ndim=1] y,
                                           np.ndarray out=None):
    """derivative of hardtanh
    out : array_like, optional
        Alternative output array in which to place the result.
    """
    it = np.nditer([y, out],
                   op_flags = [['readonly'],
                               ['writeonly', 'allocate', 'no_broadcast']])
    for w, o in it:
        if -1.0 <= w <= 1.0:
            o[...] = 1.0
        else:
            o[...] = 0.0
    return it.operands[1]

cdef np.ndarray[FLOAT_t, ndim=2] hardtanhd2d(np.ndarray[FLOAT_t, ndim=2] y,
                                             np.ndarray out=None):
    """derivative of hardtanh
    out : array_like, optional
        Alternative output array in which to place the result.
    """
    it = np.nditer([y, out],
                   op_flags = [['readonly'],
                               ['writeonly', 'allocate', 'no_broadcast']])
    for w, o in it:
        if -1.0 <= w <= 1.0:
            o[...] = 1.0
        else:
            o[...] = 0.0
    return it.operands[1]

cdef np.ndarray[FLOAT_t, ndim=1] hardtanhe(np.ndarray[FLOAT_t, ndim=1] y,
                                           np.ndarray out=None):
    """derivative of hardtanh in terms of y = hardtanh(x)
    out : array_like, optional
        Alternative output array in which to place the result.
    """
    it = np.nditer([y, out],
                   op_flags = [['readonly'],
                               ['writeonly', 'allocate', 'no_broadcast']])
    for w, o in it:
        if  w == -1.0 or w == 1.0:
            o[...] = 0.0
        else:
            o[...] = 1.0
    return it.operands[1]

cdef np.ndarray[FLOAT_t, ndim=2] hardtanhe2d(np.ndarray[FLOAT_t, ndim=2] y,
                                             np.ndarray out=None):
    """derivative of hardtanh in terms of y = hardtanh(x)
    out: array_like, optional
        Alternative output array in which to place the result.
    """
    it = np.nditer([y, out],
                   op_flags = [['readonly'],
                               ['writeonly', 'allocate', 'no_broadcast']])
    for w, o in it:
        if  w == -1.0 or w == 1.0:
            o[...] = 0.0
        else:
            o[...] = 1.0
    return it.operands[1]
