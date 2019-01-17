# distutils: language=c++

from network cimport *

cdef class SeqParameters(Parameters):

    # transitions
    cdef public np.ndarray transitions
    
cdef class SeqGradients(Gradients):

    # gradients for output variables
    cdef public np.ndarray output
    # gradients for hidden variables
    cdef public np.ndarray hidden
    cdef public np.ndarray transitions

cdef class SequenceNetwork(Network):

    cdef public np.ndarray input_sequence
    # FIXME: put in SeqVariables
    cdef public np.ndarray hidden_sequence

    # FIXME: clash with method in Network
    cdef float_t backpropagateSeq(self, sent_tags, scores, SeqGradients grads, float_t skipErr)

    cdef _backpropagate(self, SeqGradients grads)

    cdef np.ndarray[float_t,ndim=2] _calculate_delta(self, scores)

    cdef float_t _calculate_gradients_sll(self, np.ndarray[int_t] tags,
                                        SeqGradients grads,
                                        np.ndarray[float_t,ndim=2] scores,
                                        float_t skipErr)

    cdef float_t _calculate_gradients_wll(self, np.ndarray[int_t] tags,
                                        SeqGradients grads,
                                        np.ndarray[float_t,ndim=2] scores)

    cpdef np.ndarray[int_t] _viterbi(self,
                                            np.ndarray[float_t,ndim=2] scores)
