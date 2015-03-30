"""
Sequence Tagger.
"""

cimport numpy as np
from extractors cimport Converter
from networkseq cimport SeqGradients, SequenceNetwork
# use double floats
ctypedef np.double_t FLOAT_t
ctypedef np.int_t INT_t
from cpython cimport bool

cdef class Tagger(object):
        
    # feature extractor
    cdef public Converter converter
    #cdef list feature_tables

    cdef dict tags_dict
    cdef list itd
    cdef public nn # cython crashes with SequenceNetwork

    # padding stuff
    cdef np.ndarray padding_left, padding_right
    cdef public np.ndarray pre_padding, post_padding

    cpdef np.ndarray[FLOAT_t,ndim=2] _tag_sequence(self,
                                                  np.ndarray sentence,
                                                  bool train=*)

    cpdef update(self, SeqGradients grads, float learning_rate,
                 np.ndarray[INT_t,ndim=2] sentence, SeqGradients ada=*)
