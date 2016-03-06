# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = deepnl/WordsTrainer.cpp
# cython: profile=False

"""
Train a Language model.
"""

# standard
import numpy as np
import logging, sys
import time
import threading

# profiler
#import yappi

from Queue import Queue
# DEBUG
# from collections import deque
# class Queue(deque):
#     def __init__(self, maxsize=0):
#         super(Queue, self).__init__()
#     def put(self, item, wait=False):
#         super(Queue, self).append(item)
#     def get(self):
#         if len(self):
#             return super(Queue, self).popleft()
#         else:
#             return None

# for method decorations
cimport cython

# local
from math cimport *
from network cimport *
from numpy import int32 as INT
from trainer cimport Trainer
from extractors cimport Iterable
from word_dictionary import WordDictionary as WD
import utils

cdef class RandomPool(object):
    """
    An iterator returning random tokens.
    """
    # FIXME: makes sense to generate random features besides word ID?

    def __init__(self, dims, int size=1000):
        self.dims = dims        # rows in each table
        self.size = size
        self._new_pool()

    def _new_pool(self):            
        """
        Creates a pool of random feature indices, used for negative examples.
        """
        # generate 1000 indices for each table and then transpose
        # so that each row represents a token, e.g.
        # [[rnd0_t1 rnd0_tn], ..., [rnd999_t1 rnd999_tn]]
        self.pool = np.array([np.random.random_integers(0, dim - 1, self.size)
                              for dim in self.dims], dtype=INT).T
        self.current = 0

    def next(self):
        """
        Generates randomly a token for use as a negative example.
        :return: a list of token features, one for each feature table
        """
        if self.current == len(self.pool):
            self._new_pool()
        
        token = self.pool[self.current]
        self.current += 1
        
        return token

# ----------------------------------------------------------------------

cdef class LmGradients(Gradients):
    
    def __init__(self, int input_size, int hidden_size, int output_size):
        super(LmGradients, self).__init__(input_size, hidden_size, output_size)
        # gradients for negative examples
        self.input_neg = np.zeros(input_size, dtype=np.double)

    def clear(self):
        super(LmGradients, self).clear()
        self.input_neg.fill(0.0)

# ----------------------------------------------------------------------

cdef class LmNetwork(Network):

    cdef gradients(self, int slen=0):
        return LmGradients(self.input_size, self.hidden_size, self.output_size)

# ----------------------------------------------------------------------

# minimum hinge loss worth considering
cdef float minError = 1.e-5

# minimum delay between reports
cdef float report_interval = 10.0

cdef class LmTrainer(Trainer): 
    """
    Learn word representations.
    """
    
    # cdef list feature_tables
    # # data for statistics during training. 
    # cdef int total_pairs
    # cdef float error

    def __init__(self, nn, Converter converter, dict options):
        """
        :param learning_rate: initial learning rate
        :param left_cotext: left window size
        :param right_cotext: right window size
        :param hidden_size: number of hidden units
        :param output_size: number of outputs
        :param ngrams: size of ngrams to extract
        """
        super(LmTrainer, self).__init__(nn, converter, options)
        self.feature_tables = [e.table for e in converter.extractors]

    cdef _update_weights(self, worker, LmGradients grads, float remaining):
        """
        Adjust the weights along the gradients and copy them back to worker.
        :param worker: worker thread.
        :param grads: the gradients to apply.
        :param remaining: percentage reduction to apply to learning rate.
        """
        cdef Network nn = self.nn

        # decrease linearly by remaining percentage
        cdef float LR = max(0.001, self.learning_rate * remaining)

        nn.p.update(grads, LR)

        # copy new weights to worker
        worker.nn.p.copy(nn.p)

    cdef _update_embeddings(self,
                            np.ndarray[float_t] grads_input_pos,
                            np.ndarray[float_t] grads_input_neg,
                            float_t remaining,
                            np.ndarray[int_t,ndim=2] example,
                            np.ndarray[int_t] token_pos,
                            np.ndarray[int_t] token_neg):
        """
        Update the weight vectors.
        :param grads_input_pos: gradients from the positive example.
        :param grads_input_neg: gradients from the negative example.
        :param remaining: percentage used for progressively decreasing learning rate.
        :param example: the example in question.
        :param token_pos: the center token in the example.
        :param token_neg: the alternative to the center token in the example.

        These are updated separately, without locking, after each training pair.
        We rely on the fact that the words in the examples are unlikely
        to overlap, and even if that happens, operation += can be dealt
        atomically if using GPUs.
        @see http://www.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf
        F. Niu et al. 2011. Hogwild!: A Lock-Free Approach to Parallelizing
        Stochastic Gradient Descent.
        """
        # decrease linearly by remaining
        cdef float LR_0 = max(0.001, self.learning_rate * remaining)

        cdef np.ndarray[int_t] token
        cdef np.ndarray[float_t,ndim=2] table
        cdef int i, j
        cdef int start = 0, end

        for i, token in enumerate(example):
            for j, table in enumerate(self.feature_tables): # just one table
                # i-th token in the window
                # j-th feature table (there is only one)
                end = start + table.shape[1]
                deltas_pos = LR_0 * grads_input_pos[start:end]
                deltas_neg = LR_0 * grads_input_neg[start:end]
                if i == self.left_context:
                    # this is the middle position.
                    # apply negative and positive deltas to different tokens
                    table[token_pos[j]] += deltas_pos
                    table[token_neg[j]] += deltas_neg

                else:
                    # this is not the middle position. both deltas apply.
                    table[token[j]] += deltas_neg + deltas_pos
                
                start = end

    def train(self, Iterable sentences, int epochs, int report_freq,
              int threads=1, int chunk_size=100, int epoch_pairs=0):
        """
        Trains the language model on the given sentences.
        :param sentences: an iterable over sentences that can be iterated repeatedly.
        :param epochs: number of iterations over the sentences.
        :param threads: number of worker threads to use.
        :param chunk_size: size of groups of sentences assigned to each thread.
        """
        
        # TODO:
        # Use AdaGrad and initialize worker weights after warmup of one
        # iteration, as in downpour:
        # @see
        # J. Dean et al. 2010. Large Scale Distributed Deep Networks. NIPS 2012.
        # http://research.google.com/archive/large_deep_networks_nips2012.pdf

        # how often to save model
        cdef int save_period = 10000 * chunk_size

        # estimate on the maximum number of pairs from the sentences
        if epoch_pairs == 0:
            logging.info("Estimating max number of pairs")
            epoch_pairs = sum([len(sen) for sen in sentences])
        cdef float max_pairs = float(epoch_pairs * epochs)

        # perform “asynchronous stochastic gradient descent” without locking.

        # see: http://radimrehurek.com/2013/10/parallelizing-word2vec-in-python/
        # all threads access the same matrix of neural weights, and there’s no
        # locking, so they may be overwriting the same weights willy-nilly at
        # the same time.

        jobs = Queue(maxsize= 2 * threads)  # limit number of queued jobs
        lock = threading.Lock()  # for protecting shared state.

        cdef float now, start = time.time()
        # use a list, so that each thread may update the value.
        next_report = [start + report_interval] # when to report

        dims = [table.shape[0] for table in self.feature_tables]

        def train_worker(int i):
            """
            Train the model, lifting lists of sentences from the job queue.
            """
            cdef Network nn = self.nn
            # each worker thread has its own network, but they all share the
            # same converter tables (and hence feature_tables)
            worker = LmWorker(self.converter, self.learning_rate,
                              self.left_context, self.right_context,
                              dims, nn.hidden_size, ngrams=self.ngram_size)

            cdef int total_pairs
            cdef float remaining, error

            try:
                while True:
                    job = jobs.get()
                    if job is None:  # data finished, exit
                        del worker.trainer
                        return

                    # FIXME: total_pairs should be the value when job was
                    # assigned. It might have changed in the meantime.
                    total_pairs = self.total_pairs
                    remaining = 1.0 - (total_pairs / max_pairs)
                    # this also updates the input weights in the master, outside lock
                    job_pairs, error = worker._train_batch(job, remaining)

                    reporting = False
                    with lock:
                    #if True:    # DEBUG
                        # update the weights in the master, copy them back to worker
                        self._update_weights(worker, worker.grads, remaining)
                        self.avg_error.add(error)
                        self.total_pairs += job_pairs
                        now = time.time()
                        if now > next_report[0]:
                            # wait at least 10 seconds between progress reports
                            next_report[0] = now + report_interval
                            # wait longer if saving model, outside lock
                            if total_pairs and total_pairs % save_period == 0:
                                next_report[0] += 1.0
                            reporting = True

                    if reporting:
                        # guess the epoch, since threads might be working at different ones
                        epoch = total_pairs / epoch_pairs + 1
                        self._progress_report(epoch, total_pairs, len(job))

                    # save language model.
                    if total_pairs and total_pairs % save_period == 0:
                        self.saver(self)
                    #jobs.task_done() # only needed if using jobs.join()

            except Exception, e:
                logging.exception(e)
                # FIXME: process wont terminate since it is still producing jobs
                return

        # create and start the worker threads
        workers = [threading.Thread(target=train_worker, args=[i]) for i in xrange(threads)]
        for thread in workers:
            thread.daemon = True  # let the process die when main thread is killed
            thread.start()

        # start profiler
#        yappi.start()

        # fill the job queue
        for epoch in xrange(epochs):
            for job in utils.grouper(sentences, chunk_size):
                jobs.put(job, True) # block if full
        #jobs.put(None)          # DEBUG
        for _ in workers:
            jobs.put(None)  # signal termination
        # terminate
        for thread in workers:
            thread.join()
        #train_worker(0)           # DEBUG

    @cython.boundscheck(False)
    cdef np.ndarray[int_t] _extract_window(self,
                                                  np.ndarray[int_t,ndim=2] window,
                                                  np.ndarray[int_t,ndim=2] sentence,
                                                  int_t position, int_t size=1):
        """
        Extracts a window of tokens from the sentence, consisting of
        left_context tokens before :param position:,
        the ngram from :param position: to :param position+size:,
        right_context tokens after :param position+size:.
        This function takes care of adding padding as necessary.
        :param window: where to store the window.
	:param sentence: the sentence from which to extract the window.
	:param position: the start position of the center ngram.
        :param size: the size of ngram in the center of the window.
	:return: the center window token/ngram.
        """
        cdef int num_padding
        cdef np.ndarray padding
        cdef int left_context = len(self.pre_padding)
        cdef int right_context = len(self.post_padding)

        if position < left_context:
            num_padding = left_context - position
            for i in xrange(num_padding):
                np.copyto(window[i], self.pre_padding[-num_padding+i])
        else:
            num_padding = 0

        for i in xrange(left_context - num_padding):
            np.copyto(window[num_padding + i],
                      sentence[position - left_context + num_padding + i])
        if size == 1:
            ngram = sentence[position]
        else:
            # get IDs of each token
            ngramIDs = [sentence[i][0] for i in xrange(position, position + size)]
            # get the embeddings extractor
            extractor = self.converter.extractors[0]
            ngram = np.array([extractor.lookup_ngram(ngramIDs)], dtype=INT) # single feature_table

        window[left_context] = ngram

        # number of tokens in the sentence after the position
        tokens_after = len(sentence) - (position + size)
        if tokens_after < right_context:
            num_padding = right_context - tokens_after
            for i in xrange(tokens_after):
                np.copyto(window[left_context+1 + i],
                          sentence[position + size + i])
            # add padding
            for i in xrange(num_padding):
                np.copyto(window[left_context+1+tokens_after + i],
                          self.post_padding[-i])
        else:
            for i in xrange(right_context):
                np.copyto(window[left_context+1 + i],
                          sentence[position + size +i])
        return ngram

    def save(self, filename):
        """
        Saves the neural network to a file.
        It will save the weights, biases, sizes, and padding,
        as well as all converters' data (vocabulary, embeddings, etc.)
        """
        with open(filename, 'wb') as file:
            self.nn.save(file)
            self.converter.save(file)
    
    def save_vectors(self, filename):
        """
        Save vectors separately to file :param filename:.
        """
        self.converter.extractors[0].save_vectors(filename)

    @classmethod
    def load(cls, filename):
        """
        Loads the neural network from a file.
        It will load weights, biases, sizes and padding, as well as all
        converters' data.
        """
        with open(filename) as file:
            nn = Network.load(file)
            converter = Converter.load(file)
        trainer = LmTrainer.__new__(cls)
        trainer.nn = nn
        trainer.converter = converter

        return trainer
    
    def _progress_report(self, epoch, total_pairs, sent):
        """
        Reports progress in training of the network, including error and
        accuracy.
        """
        # logging.__init__() invokes acquire lock.
        print >> sys.stderr, ("Epoch: %d, pairs: %d, sent: %d, avg. error: %.3f"
                     % (epoch, total_pairs, sent, self.avg_error.mean))


cdef class LmWorker(LmTrainer):
    """
    Worker thread for learning word representations.
    """

    # # local storage
    # cdef np.ndarray input_values_pos
    # cdef np.ndarray input_values_neg
    # cdef np.ndarray hidden_values_pos
    # cdef LmGradients grads

    # # pool of random numbers (used for efficiency)
    # cdef public RandomPool random_pool

    # cdef WordsTrainer* trainer

    def __init__(self, Converter converter, float learning_rate,
                 int left_context, int right_context,
                 dims, int hidden_size, int output_size=1, int ngrams=1):

        super(LmWorker, self).__init__(converter, learning_rate,
                                       left_context, right_context,
                                       hidden_size, output_size, ngrams)

        # generate 1000 random indices at a time to save time
        # (generating 1000 integers at once takes about 10 times the time
        # for a single one)
        self.random_pool = RandomPool(dims)

        # local storage to avoid allocations:
        input_size = self.nn.input_size
        self.grads = self.nn.gradients()
        self.vars_pos = Variables(input_size, hidden_size, output_size)
        self.vars_neg = Variables(input_size, hidden_size, output_size)

        cdef int window_size = left_context + 1 + right_context
        self.example = np.empty((window_size, 1), dtype=INT)

        # build Eigen trainer, sharing arrays with Python
        self.trainer = new WordsTrainer(
            input_size, hidden_size, output_size,
            <double*>np.PyArray_DATA(self.nn.hidden_weights),
            <double*>np.PyArray_DATA(self.nn.hidden_bias),
            <double*>np.PyArray_DATA(self.nn.output_weights),
            <double*>np.PyArray_DATA(self.nn.output_bias),
            <double*>np.PyArray_DATA(self.vars_pos.input),
            <double*>np.PyArray_DATA(self.vars_neg.input),
            # grads
            <double*>np.PyArray_DATA(self.grads.hidden_weights),
            <double*>np.PyArray_DATA(self.grads.hidden_bias),
            <double*>np.PyArray_DATA(self.grads.output_weights),
            <double*>np.PyArray_DATA(self.grads.output_bias),
            <double*>np.PyArray_DATA(self.grads.input),
            <double*>np.PyArray_DATA(self.grads.input_neg),
            <int*>np.PyArray_DATA(self.example),
            window_size,
            <double*>np.PyArray_DATA(self.feature_tables[0]),
            self.feature_tables[0].shape[0],
            self.feature_tables[0].shape[1])

    @cython.boundscheck(False)
    cdef _train_batch(self, sentences, float remaining):
        """
        :param sentences: list of sentences on which to train
        :param remaining: percentage used for progressively decreasing learning rate.
        :return: the number of pairs generated and the average error
        """
        cdef int pos, pairs = 0
        cdef float error, batch_error = 0.0
        cdef np.ndarray[int_t] pos_token, neg_token
        self.grads.clear()
        cdef int left_context = len(self.pre_padding)

        for sentence in sentences:
            for position in xrange(len(sentence)):
                # extract into example the window around the given position
                pos_token = self._extract_window(self.example, sentence,
                                                 position, self.ngram_size)

                # a token is a list of feature IDs.
                # token[0] is the list with the WordDictionary index of the word.
                while True:
                    # ensure of getting a different word
                    neg_token = self.random_pool.next()
                    if neg_token[0] != pos_token[0]:
                        break

                self.converter.lookup(self.example, self.vars_pos.input)
                self.example[left_context] = neg_token
                self.converter.lookup(self.example, self.vars_neg.input)
                error = self._train_step(self.example,
                                         pos_token, neg_token,
                                         remaining)
                batch_error += error
                pairs += 1

        return pairs, batch_error/pairs
   
    cdef float _train_step(self, example, pos_token, neg_token,
                             float remaining):
        """
        Perform a training step on a single pair of examples and update weight
        vectors.
        :return: the loss.
        """
        # Eigen version
        cdef float error, LR_0
        cdef int pos_tok = pos_token[0]
        cdef int neg_tok = neg_token[0]
        with nogil:
            error = self.trainer.train_pair()
            if error > minError:
                LR_0 = max(0.001, self.learning_rate * remaining)
                # update immediately, making embeddings visible to other workers
                self.trainer.update_embeddings(<float>LR_0, pos_tok, neg_tok)
        # Python version
        # cdef float error = self._train_pair(self.vars_pos,
        #                                       self.vars_neg, grads)

        # immediately update the embeddings which are shared among
        # master and workers.
        # J. Bengio et al. BILBOWA: Fast Bilingual Distributed,
        # Representations without Word Alignments.
        # http://arxiv.org/pdf/1410.2455
        # suggest to clip gradients to [-0.1, 0.1] to improve convergence.
        # if error > minError:
        #     self._update_embeddings(grads.input, grads.input_neg, remaining,
        #                             example, pos_token, neg_token)
        return error

    @cython.boundscheck(False)
    cdef float _train_pair(self, Variables vars_pos, Variables vars_neg,
                             LmGradients grads):
        """
        Trains the network with a pair of positive/negative examples.
        :param vars_pos: variables for the positive example.
        :param vars_neg: variables for the negative example.
        :param grads: the computed gradients are accumulated here.
        :return: the hinge loss.
        """
        
        cdef Network nn = self.nn

        nn.forward(vars_pos)
        nn.forward(vars_neg)
        
        # hinge loss
        cdef float score_pos = vars_pos.output[0]
        cdef float score_neg = vars_neg.output[0]
        cdef float error = max(0.0, 1.0 - score_pos + score_neg)
        
        if error < minError:
            return error

        # Compute the gradients

        # negative gradient for the positive example is +1, for the negative
        # one is -1
        # (remember the network still has the values of the negative example) 
        
        # output gradients
        # grads.output_pos = [1]
        # grads.output_neg = [-1]
        # grads.output_bias_x = grads.output_x
        # grads.output_weights_x = grads.output_x.T x hidden_values_x

        # output_bias is left unchanged: the correction would be 0:
        # grads.output_bias += grads.output_pos + grads.output_neg

        # (output_size) x (hidden_size) = (output_size, hidden_size)
        # (1) x (hidden_size) = (1, hidden_size)
        grads.output_weights += vars_pos.hidden - vars_neg.hidden

        # hidden gradients
        # (hidden_size) * (1) x (1, hidden_size) = (hidden_size)

        grads_hidden_pos = vars_pos.hidden # reuse memory
        hardtanhe(vars_pos.hidden, grads_hidden_pos)
        grads_hidden_pos *= nn.output_weights[0]

        grads_hidden_neg = vars_neg.hidden # reuse memory
        hardtanhe(vars_neg.hidden, grads_hidden_neg)
        grads_hidden_neg *= - nn.output_weights[0]
         
        # (hidden_size) x (input_size) = (hidden_size, input_size)
        # grads.hidden_weights = grads.hidden_bias x input_values
        grads.hidden_weights += np.outer(grads_hidden_pos, vars_pos.input) +\
                                np.outer(grads_hidden_neg, vars_neg.input)

        # grads.hidden_bias = hidden_values * output_weights.dot(grads.output_bias)
        grads.hidden_bias += grads_hidden_pos + grads_hidden_neg

        # input gradients
        # These are not accumulated, since their update is immediate.
        # (hidden_size) x (hidden_size, input_size) = (input_size)
        grads.input = grads_hidden_pos.dot(nn.hidden_weights)
        grads.input_neg = grads_hidden_neg.dot(nn.hidden_weights)
        
        return error

