# -*- coding: utf-8 -*-

"""
Train a Language model
"""

# for decorations
cimport cython

# local
from network cimport *
from trainer cimport *

cdef class RandomPool:

    cdef np.ndarray pool
    cdef dims
    cdef int_t current
    cdef int_t size

cdef class LmGradients(Gradients):
    
    cdef public np.ndarray input_neg

cdef class LmTrainer(Trainer): 
    """
    Learn word representations.
    """

    cdef list feature_tables

    # data for statistics during training. 
    cdef int_t total_pairs
    
    cdef np.ndarray[int_t] _extract_window(self,
                                        np.ndarray[int_t,ndim=2] window,
                                        np.ndarray[int_t,ndim=2] sentence,
                                                  int_t position, int_t size=*)

    cdef _update_weights(self, worker, LmGradients grads, float remaining)

    cdef _update_embeddings(self,
                            np.ndarray[float_t] grads_input_pos,
                            np.ndarray[float_t] grads_input_neg,
                            float_t remaining,
                            np.ndarray[int_t,ndim=2] example,
                            np.ndarray[int_t] token_pos,
                            np.ndarray[int_t] token_neg)

# ----------------------------------------------------------------------

cdef extern from "WordsTrainer.h": # namespace "DeepNL":
    cdef cppclass WordsTrainer:
        WordsTrainer(int, int, int,
                     double*, double*, double*, double*,
                     double*, double*, double*, double*,
                     double*, double*, double*, double*,
                     int*, int,
                     double*, int, int) except +
        double train_pair() nogil
        double update_embeddings(double, int, int) nogil

cdef class LmWorker(LmTrainer): 
    """
    Worker thread for learning word representations.
    """

    # local storage
    cdef Variables vars_pos
    cdef Variables vars_neg
    cdef LmGradients grads
    cdef np.ndarray example

    # pool of random numbers (used for efficiency)
    cdef public RandomPool random_pool

    cdef WordsTrainer* trainer

    cdef _train_batch(self, sentences, float remaining)

    cdef float _train_step(self, example, pos_token, neg_token,
                             float remaining)

    cdef float _train_pair(self, Variables vars_pos, Variables vars_neg,
                             LmGradients grads)

