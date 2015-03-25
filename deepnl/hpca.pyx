# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = deepnl/HPCA.cpp
# cython: profile=False

"""
Learn word embeddings from plain text using Hellinger PCA.

See
Lebret, Rémi, and Ronan Collobert. "Word Embeddings through Hellinger PCA." EACL 2014 (2014): 482.

Author: Giuseppe Attardi
"""

import sys                      # DEBUG
import numpy as np
cimport numpy as np

# for method decorations
cimport cython

import threading

from scipy.linalg.lapack import ssyevr
from numpy.linalg import svd

import logging
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------

ctypedef np.float32_t FLOAT_t

cdef extern from "HPCA.h" namespace "hpca" nogil:
    object cooccurrence_matrix(char* corpus, char* vocabFile, int top, int window)

# ----------------------------------------------------------------------

def cooccurrences(char* corpus, char* vocabFile, unsigned top, unsigned window):
    """
    Compute the cooccurrence matrix on a corpus of sentences, using as
    vocabulary the list of words present in :param vocabFile: and as
    context words the first :param top: words in the vocabulary.
    :param corpus: file containing text.
    """
    return cooccurrence_matrix(corpus, vocabFile, top, window)

cpdef np.ndarray[FLOAT_t,ndim=2] fit(np.ndarray[FLOAT_t,ndim=2] dm,
                                     int n_components, bint covariance=False):
    """
    Compute SVD on :param dm:.
    :param covariance: use covariance matrix.
    :return: the representation of dm reduced to :param n_components: dimensions.
    """

    cdef np.ndarray[FLOAT_t,ndim=2] cov
    cdef int cols = dm.shape[1]

    if covariance:
        # use lapack SSYEVR
        # ask just for the largest eigenvalues
        # (cols x rows) (rows x cols) = (cols x cols)
        cov = dm.T.dot(dm)
        w, z, info = ssyevr(cov, range='I', il=cols-n_components+1, overwrite_a=1)
        return dm.dot(z)
    else:
        # use lapack _gesdd
        u, s, v = svd(dm, full_matrices=False)
        # v = (cols x cols), rows are eigenvectors
        # v.T = (cols x cols), cols are eigenvectors
        # (rows x cols) (cols x comp) = (rows x comp)
        return dm.dot(v.T[:,:n_components])
        # alternative using scipy.sparse.linalg.svds
        # u, s, v = svds(dm, n_components)
