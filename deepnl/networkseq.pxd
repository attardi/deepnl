
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

    # FIXME: cant share a global?
    cdef public FLOAT_t skipErr

    cdef public np.ndarray input_sequence
    # FIXME: put in SeqVariables
    cdef public np.ndarray hidden_sequence

    # FIXME: clash with method in Network
    cdef float backpropagateSeq(self, sent_tags, scores, SeqGradients grads)

    cdef _backpropagate(self, SeqGradients grads)

    cdef np.ndarray[FLOAT_t,ndim=2] _calculate_delta(self, scores)

    cdef float _calculate_gradients_sll(self, np.ndarray[INT_t,ndim=1] tags,
                                        SeqGradients grads,
                                        np.ndarray[FLOAT_t,ndim=2] scores)

    cdef float _calculate_gradients_wll(self, np.ndarray[INT_t,ndim=1] tags,
                                        SeqGradients grads,
                                        np.ndarray[FLOAT_t,ndim=2] scores)

    cpdef np.ndarray[INT_t,ndim=1] _viterbi(self,
                                            np.ndarray[FLOAT_t,ndim=2] scores)
