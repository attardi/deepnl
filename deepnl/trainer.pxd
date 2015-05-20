
from cpython cimport bool

cimport numpy as np
from network cimport Network, Gradients
from extractors cimport Converter
from networkseq cimport SeqGradients

ctypedef np.int_t INT_t

cdef class MovingAverage(object):

    cdef float mean
    cdef float variance
    cdef unsigned count

    cdef add(self, float v)

cdef class Trainer(object):

    cdef readonly Network nn
    # feature extractor
    cdef public Converter converter
    cdef np.ndarray pre_padding, post_padding
    # size of ngrams
    cdef int ngram_size
    cdef public float learning_rate
    cdef public object saver
    cdef int train_items, skips
    cdef float accuracy
    cdef readonly MovingAverage avg_error
    cdef public bool verbose

    cdef float _validate(self, list sentences, labels, int idx)

    cpdef update(self, Gradients grads, float learning_rate,
                 np.ndarray[INT_t,ndim=2] sentence, Gradients ada=*)

cdef class TaggerTrainer(Trainer):

     cdef dict tags_dict
     #cdef Tagger tagger # FIXHIM: crashes Cython compiler
     cdef readonly object tagger


