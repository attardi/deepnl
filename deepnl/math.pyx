# distutils: language = c++

cimport numpy as np
import numpy as np

# ----------------------------------------------------------------------
# Math functions

cdef np.ndarray[float_t] softmax(np.ndarray[float_t] a, np.ndarray out=None):
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

cdef np.ndarray[float_t, ndim=2] softmax2d(np.ndarray[float_t, ndim=2] a, int_t axis=0, np.ndarray out=None):
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

cdef float_t logsumexp(np.ndarray[float_t] a):
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

cdef np.ndarray[float_t] logsumexp2d(np.ndarray[float_t,ndim=2] a, int_t axis=0):
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

cdef np.ndarray[float_t] tanh(np.ndarray[float_t] weights,
                                      np.ndarray out=None):
    """Hyperbolic tangent.
    out : array_like, optional
        Alternative output array in which to place the result.
    """
    if out is None:
        out = np.empty_like(weights)
    np.tanh(weights, out)
    return out

cdef np.ndarray[float_t] tanhe(np.ndarray[float_t] y,
                                       np.ndarray out=None):
    """Hyperbolic tangent.
    out : array_like, optional
        Alternative output array in which to place the result.
    """
    if out is None:
        out = np.empty_like(y)
    out[:] = 1 - y**2
    return out

cdef np.ndarray[float_t] hardtanh(np.ndarray[float_t] y,
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

cdef np.ndarray[float_t] hardtanhd(np.ndarray[float_t] y,
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

cdef np.ndarray[float_t, ndim=2] hardtanhd2d(np.ndarray[float_t, ndim=2] y,
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

cdef np.ndarray[float_t] hardtanh_back(np.ndarray[float_t] y,
                                               np.ndarray[float_t] grads,
                                               np.ndarray[float_t] grads_in):
    """backward of hardtanh in terms of y = hardtanh(x)
    Propagates the output gradients to the input, by multiplying with the
    derivative p hardtanh.
    grads:      gradients of output.
    grads_in :  output array in which to place the result.
    """
    it = np.nditer([y, grads, grads_in],
                   op_flags = [['readonly'], ['readonly'], ['writeonly']])
    for w, g, o in it:
        if w == -1.0 or w == 1.0:
            o[...] = 0.0
        else:
            o[...] = g[...]
    return grads_in

cdef np.ndarray[float_t, ndim=2] hardtanh_back2d(np.ndarray[float_t, ndim=2] y,
                                                 np.ndarray[float_t, ndim=2] grads_out,
                                                 np.ndarray[float_t, ndim=2] grads_in):
    """derivative of hardtanh in terms of y = hardtanh(x)
    Propagates the output gradients to the input, by multiplying with the
    derivative of hardtanh.
    grads_out:  gradients of output.
    grads_in:   array in which to place the result.
    """
    it = np.nditer([y, grads_out, grads_in],
                   op_flags = [['readonly'], ['readonly'], ['writeonly']])
    for w, gout, gin in it:
        if w == -1.0 or w == 1.0:
            gin[...] = 0.0
        else:
            gin[...] = gout[...]
    return grads_in

cdef np.ndarray[float_t] hardtanhe(np.ndarray[float_t] y,
                                           np.ndarray out=None):
    """derivative of hardtanh in terms of y = hardtanh(x)
    out : array_like, optional
        Alternative output array in which to place the result.
    """
    it = np.nditer([y, out],
                   op_flags = [['readonly'],
                               ['writeonly', 'allocate', 'no_broadcast']])
    for w, o in it:
        if w == -1.0 or w == 1.0:
            o[...] = 0.0
        else:
            o[...] = 1.0
    return it.operands[1]

cdef np.ndarray[float_t, ndim=2] hardtanhe2d(np.ndarray[float_t, ndim=2] y,
                                             np.ndarray out=None):
    """derivative of hardtanh in terms of y = hardtanh(x)
    out: array_like, optional
        Alternative output array in which to place the result.
    """
    it = np.nditer([y, out],
                   op_flags = [['readonly'],
                               ['writeonly', 'allocate', 'no_broadcast']])
    for w, o in it:
        if w == -1.0 or w == 1.0:
            o[...] = 0.0
        else:
            o[...] = 1.0
    return it.operands[1]
