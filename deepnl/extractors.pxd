# -*- coding: utf-8 -*-
# distutils: language=c++

"""
Feature extractors.
"""

cimport numpy as np

from network cimport float_t, int_t
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
    cdef readonly list fields

    cdef np.ndarray[int_t] get_padding_left(self)
    cdef np.ndarray[int_t] get_padding_right(self)

    cpdef int_t size(self)

    cpdef np.ndarray[int_t,ndim=2] convert(self, list sent)

    cpdef np.ndarray[float_t] lookup(self,
                                     np.ndarray[int_t,ndim=2] sentence,
                                     np.ndarray out=*)

    cpdef adaGradInit(self, float_t adaEps)

    cpdef update(self, np.ndarray[float_t] grads, np.ndarray[int_t,ndim=2] sentence,
    	  	 float_t learning_rate)

cdef class Extractor(object):

    cdef readonly dict dict
    cdef readonly np.ndarray table
    cdef readonly np.ndarray adaGrads

    cpdef int_t size(self)

    cpdef adaGradInit(self, float_t adaEps)

    cpdef int_t get_padding_left(self)
    cpdef int_t get_padding_right(self)

    cpdef extract(self, list tokens, int_t field)

cdef class Embeddings(Extractor):
    pass

cdef class CapsExtractor(Extractor):
    pass

cdef class AffixExtractor(Extractor):
    cdef bool lowcase

cdef class SuffixExtractor(AffixExtractor):
    pass

cdef class PrefixExtractor(AffixExtractor):
    pass

cdef class GazetteerExtractor(Extractor):
    cdef bool lowcase
    cdef bool noaccents

cdef class AttributeExtractor(Extractor):
    pass
