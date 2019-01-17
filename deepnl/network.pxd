# distutils: language=c++

cimport numpy as np

# Use double floats
ctypedef double float_t
# Use 32bit int
ctypedef int int_t
# dtype('int32')
from numpy import int32 as INT

cdef class Variables(object):
    """Visible and hidden variables.
    Unique to thread"""
    
    cdef public np.ndarray input, hidden, output

cdef class Parameters(object):
    """
    Network parameters: weights and biases.
    Shared by threads.
    """

    cdef public np.ndarray hidden_weights, hidden_bias
    cdef public np.ndarray output_weights, output_bias

    cdef copy(self, Parameters p)
    # cpdef since it is called with super
    cpdef update(self, Gradients grads, float_t learning_rate,
                 Parameters ada=*)

cdef class Gradients(Parameters):

    # gradients for input variables
    cdef public np.ndarray input

cdef class Network(object):
    
    cdef public Parameters p

    # sizes (public for loading)
    cdef public int input_size, hidden_size, output_size
    
    # function to save periodically
    cdef public object saver

    cdef variables(self, int slen=*)
    cdef gradients(self, int slen=*)
    cdef parameters(self)

    cpdef forward(self, Variables vars)

    cdef float_t backpropagate(self, int y, Variables vars, Gradients grads)

    # cpdef since used with super
    cpdef update(self, Gradients grads, float_t learning_rate, Parameters ada=*)
