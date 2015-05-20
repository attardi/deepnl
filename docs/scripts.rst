.. _scripts:

==================
Standalone Scripts
==================

:mod:`nlpnet` includes standalone scripts that may be called from a command line. They are 
copied to the `scripts` subdirectory of your Python installation, which can be included 
in the system PATH variable. There are three such scripts:

**nlpnet-train**
  Script to train a new model or further train an existing one.

**nlpnet-test**
  Script to measure the performance of a model against a gold data set.

**nlpnet-tag**
  Script to call a model and tag some given text.

Each of them is explained below.

.. contents::  
  :local:  
  :depth: 1  


nlpnet-tag
==========

This is the simplest :mod:`nlpnet` script. It simply runs the system for a given text input. 
It should be called with the following syntax:

.. code-block:: bash

    $ nlpnet-tag.py TASK DATA_DIRECTORY

Where ``TASK`` is either ``pos`` or ``srl`` and ``DATA_DIRECTORY`` is the directory with the
trained models. It has also the following command line options:

-v  Verbose mode
--no-repeat  Forces the classification step to avoid repeated argument labels (SRL only).

For example:

.. code-block:: bash

    $ nlpnet-tag.py pos /path/to/nlpnet-data/
    O rato roeu a roupa do rei de Roma.
    O_ART rato_N roeu_V a_ART roupa_N do_PREP+ART rei_N de_PREP Roma_NPROP ._PU

Or with semantic role labeling:

.. code-block:: bash

    $ nlpnet-tag.py srl /path/to/nlpnet-data/
    O rato roeu a roupa do rei de Roma.
    O rato roeu a roupa do rei de Roma .
    roeu
        A1: a roupa do rei de Roma
        A0: O rato
        V: roeu

The first line was typed by the user, and the second one is the result of tokenization.



nlpnet-train
============

There are a lot of training parameters that can be supplied to :mod:`nlpnet`. Some of them depend
on the task that the network is being trained for, as the type of network can be a simple MLP
for POS tagging and a convolutional network for SRL.

General Options
---------------

These options can be used in either POS or SRL training.

-w NUMBER  The size of the word window. For SRL, the supplied model used 3, and for POS, 5. It is important to have a reasonably large window in POS so the tagger can analyze the context.
-n NUMBER  Number of hidden neurons.
-f NUMBER  Generates feature vectors randomly with the given number of dimensions for words. Ignore it if you supply pre-initialized representations.
--load_features  Loads the features vectors representing words. The file containing the data must be set in config.py and be in the data/ directory. Nlpnet uses numpy files for storing representations as 2-dimensional arrays.
-e NUMBER  Number of epochs to train the network.
-l NUMBER  The learning rate for network weights.
--lf NUMBER  The learning rate for features (including extra features like the ones from ``--caps``).
--lt NUMBER  The learning rate for the tag transition scores.
--caps NUMBER  Include capitalization as a feature. If a number is given, determine the number of features (default 5).
--suffix NUMBER  Same as ``--caps``, but for suffixes. It will search a file named suffixes.txt in the data/ directory, and read each line as suffix.
-a NUMBER  Stop training when the network achieves this accuracy. Useful to avoid divergence when the learning rate is high.
-v  Verbose mode, it will output more information about what is happening internally.
--load_network  Loads a previously saved network. The file name must be set in config.py and be in the data/ directory. 
--task TASK  Task to train for. It must be either ``srl`` or ``pos``.
--data DIRECTORY  The directory containing the model files. If a new model is being trained, everything is saved to that dir.
--gold FILE  A file containing the gold data used for training.

Data files must be in the format used by :mod:`nlpnet`. A POS file must have one sentence per line, each sentence containing tokens in the format ``token_tag`` and separated by whitespace. SRL files must be in the `CoNLL format`_.

.. _`CoNLL format`: https://ufal.mff.cuni.cz/conll2009-st/task-description.html#Dataformat


SRL
---

-c NUMBER  Number of neurons in the convolution layer.
--pos NUMBER  Uses POS as a feature. Currently, it must read the tags from the training data. Works same as --caps.
--chunk NUMBER  Uses syntactic chunks as a feature. Same as --pos.
--use_lemma  Reads word lemmas instead of surface forms. It needs to read them from the training data.
--id  Train for argument boundary identification only.
--class  Train for previously identified argument classification only. (if neither this or ``--id`` is supplied, trains a network that does both in a single step)
--pred  Train for predicate recognizing only.
--max_dist NUMBER  The maximum distance (to predicates and target words) to have an own feature vector. Any distance greater than this will be mapped to a single vector.
--target_features NUMBER  Number of features for vectors representing distance to the target word.
--pred_features NUMBER  Same as ``--target_features`` for the predicate.


nlpnet-test
===========

This script is much simpler. It evaluates the system performance against a gold standard. 

General options
---------------

The arguments below are valid for both tasks.

--task TASK  Task for which the network should be used. Either ``pos`` or ``srl``.
-v  Verbose mode
--gold FILE  File with gold standard data
--data DIRECTORY  Directory with trained models

POS
---

--oov FILE  Analyze performance on the words described in the given file.

The ``--oov`` option requires a UTF-8 file containing one word per line. Actually, this option
is not exclusive for OOV (out-of-vocabulary) words, but rather any word list you
want to evaluate.

SRL
---

SRL evaluation is performed in different ways, depending on whether it is aimed at
argument identification, classification, predicate detection or all of them.
In the future, there may be a more standardized version for this test.

--id  Evaluate only argument identification (SRL only). The script will output the score.
--class  Evaluate only argument classification (SRL only). The script will output the score.
--preds  Evaluate only predicate identification (SRL only). The script will output the score.
--2steps  Execute SRL with two separate steps. The script will output the results in CoNLL format.
--no-repeat  Forces the classification step to avoid repeated argument labels (2 step SRL only)
--auto-pred  Determines SRL predicates automatically. Only used when evaluating the full process (identification + classification)

The CoNLL output can be evaluated against a gold file using the official SRL eval script (see http://www.lsi.upc.edu/~srlconll/soft.html).


