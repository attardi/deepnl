# -*- coding: utf-8 -*-
"""
Feature extractors.
"""

cimport numpy as np

# use double floats
ctypedef np.double_t FLOAT_t
ctypedef np.int_t INT_t
from cpython cimport bool

# ----------------------------------------------------------------------

cdef class Iterable:
    """
    ABC for classes that provide the __iter__() method.
    """

# ----------------------------------------------------------------------

cdef class Converter(Iterable):
    """
    Interface to the extractors.
    Extracts features from a sentence and converts them into a list of feature
    vectors in feature space.
    """
    
    cdef readonly list extractors

    cdef np.ndarray[INT_t,ndim=1] get_padding_left(self)

    cdef np.ndarray[INT_t,ndim=1] get_padding_right(self)

    cpdef int size(self)

    cpdef np.ndarray[INT_t,ndim=2] convert(self, list sent)

    cpdef np.ndarray[FLOAT_t,ndim=1] lookup(self,
                                            np.ndarray[INT_t,ndim=2] sentence,
                                            np.ndarray out=*)
    cpdef update(self, np.ndarray[INT_t,ndim=2] sentence,
                 np.ndarray[FLOAT_t,ndim=1] grads)

cdef class Extractor(object):

    cdef readonly dict dict

    cdef readonly np.ndarray table

    cpdef int size(self)

cdef class Embeddings(Extractor):
    pass

cdef class CapsExtractor(Extractor):
    pass

cdef class AffixExtractor(Extractor):
    pass

cdef class SuffixExtractor(AffixExtractor):
    pass

cdef class PrefixExtractor(AffixExtractor):
    pass

cdef class GazetteerExtractor(Extractor):
    cdef bool lowcase
