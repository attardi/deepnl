============
Introduction
============

This documents covers the basics for installing and using :mod:`deepnl`. 

Installation
------------

:mod:`deepnl` can be downloaded from the Python package index at https://pypi.python.org/pypi/deepnl/ or installed with

.. code-block:: bash

    pip install deepnl

See the `Dependencies`_ section below for additional installation requirements.
    
Dependencies
~~~~~~~~~~~~

:mod:`deepnl` requires and numpy_.

For development use you will also need Cython_, which is used to generate C
extensions and run faster.

For simple installation, you dont't need it, since the generated ``.c`` files are already provided with :mod:`deepnl`, but you will need a C compiler.

.. _numpy: http://www.numpy.org
.. _Cython: http://cython.org
.. _setuptools: http://pythonhosted.org/setuptools/

Brief explanation
-----------------

Here is a brief exaplanation about how stuff works in the internals of
:mod:`deepnl` (*you don't need to know it to use this library*).
For additional details on the technique, refer to the articles in the index page or about the SENNA system.

Two types of neural networks are available: a common MLP (multilayer
perceptron) and a convolutional one. 
The former is used for training a POS tagger, and a NER tagger.
Basically, the common MLP examines
word windows, outputs a score for assigning each tag to each word, and then determines 
the tags using the Viterbi algorithm (which is essentially picking the best combination from network
scores and tag transition scores).

During training, adjustments are made to the network connections, word representations and 
the tag transition scores. Their learning rates may be set separately, although the best
results seem to arise when all three have the same value.

The convolutional network can be used to train a Semantic Role Labeler (SRL).
In order to output a score for each word, it examines the whole sentence. It does so by picking a word window at a time and forwarding it to a convolution layer.
This layer stores in each of its neurons the biggest value found so far.
After all words have been examined, the convolution layer forwards its output like a usual MLP network.
Then, it works like the previous model: the network outputs scores for each word/tag combination,
and a Viterbi search is performed.

In the convolution layer, the values found by each neuron may come from different words, i.e., each neuron stores
its maximum independently from the others. This is particularly complex during training, because 
neurons must backpropagate their error only to the word window that yielded their stored value.

One doesn't need to worry about the details concerning the neural networks
when using the standalone scripts provided in the ``bin`` directory:
- ``dl-words.py``,
- ``dl-sentiwords.py``,
- ``dl-ner.py``,
- ``dl-pos.py``.

However, they are available to play with in the :ref:`network` module.

Basic usage
-----------

:mod:`deepnl` can be used both as a Python library or by its standalone scripts. The basic library API is explained below.
See also :ref:`scripts`.

Library usage
~~~~~~~~~~~~~

You can use :mod:`deepnl` as a library in Python code as follows:

.. code-block:: python

    >>> import deepnl
    >>> tagger = deepnl.PosTagger('modelFile')
    >>> tagger.tag('O rato roeu a roupa do rei de Roma.')
    [[(u'O', u'ART'), (u'rato', u'N'), (u'roeu', u'V'), (u'a', u'ART'), (u'roupa', u'N'), (u'do', u'PREP+ART'), (u'rei', u'N'), (u'de', u'PREP'), (u'Roma', u'NPROP'), (u'.', 'PU')]]

In the example above, ``'modelFile'`` must be the path to the file containing
the trained POS model.

The curently available taggers are:
- ``PosTagger`` and
- ``NerTagger``.

Both taggers expect sentences consisting of a list of tokens.

The output is printed in TSV format, with one token per line, and each line
contains:

word<tab>tag

with an empty line separating sentences.


