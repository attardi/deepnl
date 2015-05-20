
from network cimport *

cdef class ConvVariables(Variables):
    """Visible and hidden variables.
    Unique to thread"""
    
    cdef public np.ndarray hidden2
    # convolution layer
    cdef readonly np.ndarray conv
    # maximum convolution indices
    cdef readonly np.ndarray tmax

cdef class ConvParameters(Parameters):
    """
    Network parameters: weights and biases.
    Shared by threads.
    """

    # the second hidden layer
    cdef readonly np.ndarray hidden2_weights, hidden2_bias

cdef class ConvGradients(ConvParameters):
    
    cdef public np.ndarray conv

cdef class ConvolutionalNetwork(Network):

    # FIXME: we must add it here, since Network inherits from Parameters
    # FIXME: use dependency injection for Parameters
    # the second hidden layer
    cdef readonly np.ndarray hidden2_weights, hidden2_bias
    cdef readonly int hidden2_size

    cpdef run(self, Variables vars)

    cdef np.ndarray[FLOAT_t,ndim=1] predict(self,
                                            np.ndarray[INT_t,ndim=2] sentence,
                                            vars,
                                            bool train=*)

    cdef float backpropagate(self, int y, Variables vars, Gradients grads)

    cpdef update(self, Gradients grads, float learning_rate,
                 Gradients ada=*)
