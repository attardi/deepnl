
from network cimport *

cdef class SeqGradients(Gradients):

    cdef public np.ndarray output
    cdef public np.ndarray transitions

cdef class SequenceNetwork(Network):

    # transitions
    cdef public np.ndarray transitions
    
    cdef public np.ndarray input_sent_values, hidden_sent_values, layer2_sent_values

    cdef _backpropagate(self, SeqGradients grads)

    cdef _update(self, SeqGradients grads, float learning_rate)

    cpdef bool _calculate_gradients_sll(self, np.ndarray[INT_t,ndim=1] tags,
                                       SeqGradients grads,
                                       np.ndarray[FLOAT_t,ndim=2] scores)

    cpdef bool _calculate_gradients_wll(self, np.ndarray[INT_t,ndim=1] tags,
                                        SeqGradients grads,
                                        np.ndarray[FLOAT_t,ndim=2] scores)

    cpdef np.ndarray[INT_t,ndim=1] _viterbi(self,
                                            np.ndarray[FLOAT_t,ndim=2] scores)
