"""
Sequence Tagger.
"""

cimport numpy as np
from extractors cimport Converter
from networkseq cimport SeqGradients, SequenceNetwork
from network cimport float_t, int_t
from cpython cimport bool

cdef class Tagger(object):
        
    # feature extractor
    cdef public Converter converter

    cdef readonly dict tag_index         # tag ids
    cdef readonly list tags              # list of tags
    cdef public nn # cython crashes with SequenceNetwork

    # padding stuff
    cdef public np.ndarray pre_padding, post_padding

    cpdef list tag(self, list tokens)

    cpdef np.ndarray[float_t,ndim=2] _tag_sequence(self,
                                                   np.ndarray sentence,
                                                   bool train=*)

