
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

    # public to enable loading
    cdef public Network nn
    # feature extractor
    cdef public Converter converter
    cdef public np.ndarray pre_padding, post_padding
    cdef public object saver
    cdef int total_items, epoch_items, epoch_hits, skips
    # data for statistics
    cdef float error, accuracy
    cdef readonly MovingAverage avg_error

    # options
    cdef public bool verbose

    # size of ngrams
    cdef public int ngram_size

    # training parameters
    cdef public float learning_rate
    # turned into globals since can't be static variables
    # cdef public float l1_decay
    # cdef public float l2_decay
    # cdef public float momentum
    # cdef public float ro # used in AdaDelta
    # cdef public float eps # used in AdaGrad

    cdef public float skipErr

    cdef float _validate(self, list sentences, labels, int idx)

    cpdef update(self, Gradients grads, float learning_rate,
                 np.ndarray[INT_t,ndim=2] sentence, Gradients ada=*)

cdef class TaggerTrainer(Trainer):

     cdef dict tags_dict
     #cdef Tagger tagger # FIXHIM: crashes Cython compiler
     cdef readonly object tagger


