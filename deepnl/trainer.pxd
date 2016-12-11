
from cpython cimport bool
cimport numpy as np

# local
from network cimport Network, Parameters, Gradients, float_t, int_t
from extractors cimport Converter
from networkseq cimport SeqGradients

cdef class MovingAverage(object):

    cdef float_t mean
    cdef float_t variance
    cdef unsigned count

    cdef add(self, float_t v)

cdef class Trainer(object):

    # public to enable loading
    cdef public Network nn
    # feature extractor
    cdef public Converter converter
    cdef public np.ndarray pre_padding, post_padding
    cdef public object saver
    cdef int total_items, epoch_items, epoch_hits, skips
    # data for statistics
    cdef float_t error, accuracy
    cdef readonly MovingAverage avg_error

    # options
    cdef public bool verbose

    # size of ngrams
    cdef public int ngram_size

    # training parameters
    cdef public float_t learning_rate
    cdef float_t adaEps
    cdef float_t adaRo
    cdef float_t l1_decay
    cdef float_t l2_decay
    cdef float_t momentum
    cdef float_t skipErr
    cdef Parameters ada

    cdef float_t _validate(self, list sentences, labels, int idx)

    cpdef update(self, Gradients grads, np.ndarray[int_t,ndim=2] sentence)

cdef class TaggerTrainer(Trainer):

     cdef dict tags_dict
     #cdef Tagger tagger # FIXHIM: crashes Cython compiler
     cdef readonly object tagger


