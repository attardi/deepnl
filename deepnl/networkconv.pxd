
from network cimport *

cdef class ConvVariables(Variables):
    """Visible and hidden variables.
    Unique to thread"""
    
    cdef public np.ndarray hidden2
    # convolution layer
    cdef readonly np.ndarray conv

cdef class ConvParameters(Parameters):
    """
    Network parameters: weights and biases.
    Shared by threads.
    """

    # the second hidden layer
    cdef public np.ndarray hidden2_weights, hidden2_bias

    cpdef update(self, Gradients grads, float learning_rate, Gradients ada=*)

cdef class ConvGradients(Gradients):
    
    cdef public np.ndarray hidden2_weights, hidden2_bias
    cdef readonly np.ndarray conv

cdef class ConvolutionalNetwork(Network):

    cdef public int hidden2_size
    cdef public int pool_size

    cdef np.ndarray[FLOAT_t,ndim=1] predict(self, list tokens)

    cdef float backpropagate(self, int y, Variables vars, Gradients grads)
