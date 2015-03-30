
from network cimport *

cdef class ConvolutionalNetwork(Network):
    
    # transition and distance feature tables
    cdef readonly np.ndarray target_dist_table, pred_dist_table
    cdef readonly np.ndarray target_dist_weights, pred_dist_weights
    cdef readonly int target_dist_offset, pred_dist_offset
    cdef readonly np.ndarray target_dist_lookup, pred_dist_lookup
    cdef readonly np.ndarray target_dist_deltas, pred_dist_deltas
    
    # the second hidden layer
    cdef readonly int hidden2_size
    cdef readonly np.ndarray hidden2_weights, hidden2_bias
    cdef readonly np.ndarray hidden2_values
    cdef readonly np.ndarray layer3_sent_values
    
    # in training, we need to store all the values calculated by each layer during 
    # the classification of a sentence. 
    cdef readonly np.ndarray hidden2_sent_values
    
    # maximum convolution indices
    cdef readonly np.ndarray max_indices
    
    # number of targets (all tokens in a sentence or the provided arguments)
    # and variables for argument classifying
    cdef int num_targets
    cdef bool only_classify
    
    # for faster access 
    cdef int half_window, features_per_token
    
    # the convolution gradients 
    cdef np.ndarray hidden_gradients, hidden2_gradients
    cdef np.ndarray input_deltas

    cdef np.ndarray[FLOAT_t,ndim=2] _tag_sequence_conv(self, np.ndarray[INT_t,ndim=2] sentence,
                      list tags=*,
                      np.ndarray predicates=*, list arguments=*, 
                      bool logprob=*, bool allow_repeats=*)

    cdef np.ndarray argument_distances(self, positions, argument)

    cdef _backpropagate_conv(self, sentence)

    cdef np.ndarray[INT_t,ndim=1] _viterbi_conv(self, np.ndarray[FLOAT_t,ndim=2] scores, bool allow_repeats=*)
