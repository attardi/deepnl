************************************************************
``deepnl`` --- Deep Learning for Natural Language Processing
************************************************************

``deepnl`` is a Python library for Natural Language Processing tasks based on
a Deep Learning neural network architecture.

The library currently provides tools for performing part-of-speech tagging,
Named Entity tagging and Semantic Role Labeling.

``deepnl`` also provides code for creating *word embeddings* from text, using
either the Language Model approach by [Collobert11]_, or Hellinger PCA,
as in [Lebret14]_.

It can also create *sentiment specific word embeddings* from a corpus of
annotated Tweets.

If you use ``deepnl``, please cite [Attardi]_ in your publications.

**WARNING**. There has been a change in file format for models since version 1.3.14.
You will have to retrain them to use with later versions.

Installation
===========

Download the code or clone the repository on your machine with:

   $ git clone https://github.com/attardi/deepnl.git
   
Ensure that you have the dependencies mentioned below, then proceed to the build process described below.

Dependencies
------------

``deepnl`` requires numpy_ and Eigen_.

A C++ compiler is also needed for compiling the C++ extensions it uses,
produced with Cython_.
The generated ``.cpp`` files are already provided with ``deepnl``, but you
will need Cython_ if you want to develop or modify the C++ extensions.

Build
-----

To compile the library, run::

   $ python2 setup.py build

This will invoke the C++ compiler to compile the code on your platform.

You can run the scripts directly from the ``bin`` directory, or you can
install them by calling::

   $ sudo python setup.py install

If Cython gets invoked and raises error, force an update on the file
timestamps, with:

   $ touch deepnl/*.cpp

Basic usage
===========

``deepnl`` can be used both as a Python library or through command line scripts.

Library usage
-------------

You can use ``deepnl`` as a library in Python code as follows, where
``filename`` is the name of the file containing the model produced through training:

.. code-block:: python

    >>> from deepnl.tagger import Tagger
    >>> tagger = Tagger.load(open(filename))
    >>> sent = 'The quick brown fox jumped over the lazy dog .'
    >>> tagger.tag_sequence(sent.split(), return_tokens=True)
    [[(u'The', u'DT'), (u'quick', u'JJ'), (u'brown', u'JJ'), (u'fox', u'NN'), (u'jumped', u'VBD'), (u'over', u'IN'), (u'the', u'DT'), (u'lazy', u'JJ'), (u'dog', u'NN'), (u'.', '.')]]

Class ``Tagger`` is a generic interface for sequence taggers and provides a
method ``tag_sequence`` for tagging a sentence.
A sentence is represented as a list of tokens.

Class ``Tagger`` can be used directly for performing POS tagging.
Two specializations are provided: ``NerTagger`, for Named Entity tagging and
``SrlTagger`` for Semantic Role Labeling.

The output of ``tag_sequence`` is normally a list of tuples, representing
tokens with their associated tags. In the case of POS tagging, the tags are
just the POS tags of each token; in case of ``NerTagger`` the tags are in
``IOB`` notation for representing subsequences, while in the case of
``SrlTagger`` the output is more complex.


Standalone scripts
------------------

``deepnl`` provides scripts for tagging text or training new models.

They are present in the `bin` subdirectory where you downloaded the code.
If you did not install them, you can invoke them directly from there.

Call them with option ``-h`` or ``--help`` to obtain details on their usage.

The scripts expect tokenized input, one token per line, with an empty line to
separate sentences.

When training, the token attributes are supplied in TSV (tab separated values) format.
Here is an example of POS tagging, using a previously trained model from file ``pos.dnn``:

.. code-block:: bash

    $ dl-pos.py pos.dnn
    The
    quick
    brown
    fox
    jumped
    over
    the
    lazy
    dog
    .

    The DT
    quick JJ
    brown JJ
    fox NN
    jumped VBD
    over IN
    the DT
    lazy JJ
    dog NN
    . .

Word Embeddings
===============

The command ``dl-words.py`` allows creating word embeddings from a language
model built from a plain text corpus, properly tokenized.

The command ``dl-words-pca.py`` allows creating word embeddings from a
language model built from a plain text corpus, with the technique of Hellinger
PCA.

The command ``dl-sentiwords.py`` allows creating *sentiment specific word
embeddings* from a corpus of annotated Tweets.


Benchmarks
==========

The NER tagger replicates the performance of SENNA_ in the CoNLL 2003 benchmark.

The CoNLL-2003 shared task data can be downloaded from
http://www.cnts.ua.ac.be/conll2003/ner/.

The train and test data must be cleaned and converted to the more recent IOB2
notation, by calling:

.. code-block:: bash

    sed '/-DOCSTART-/,+1d' train | bin/toIOB.py | cut -f 1,2,4 > train.iob
    sed '/-DOCSTART-/,+1d' testa | bin/toIOB.py | cut -f 1,2,4 > testa.iob
    sed '/-DOCSTART-/,+1d' testb | bin/toIOB.py | cut -f 1,2,4 > testb.iob
    cat train.iob testa.iob > train+dev.iob

Assuming that the SENNA distribution is in directory ``senna``, the embeddings
and vocabulary from SENNA can be used:

.. code-block:: bash

   cp -p senna/embeddings/embeddings.txt vectors.txt
   cp -p senna/hash/words.lst vocab.txt

The gazetters from SENNA can be used to produce a single entity list as follows:

.. code-block:: bash

    iconv -f ISO-8859-1 -t UTF-8 < senna/hash/ner.loc.lst | awk '{printf "LOC\t%s\n", $$0}'  > eng.list
    iconv -f ISO-8859-1 -t UTF-8 < senna/hash/ner.misc.lst | awk '{printf "MISC\t%s\n", $$0}' >> eng.list
    iconv -f ISO-8859-1 -t UTF-8 < senna/hash/ner.org.lst | awk '{printf "ORG\t%s\n", $$0}' >> eng.list
    iconv -f ISO-8859-1 -t UTF-8 < senna/hash/ner.per.lst | awk '{printf "PER\t%s\n", $$0}' >> eng.list

You also need the list of suffixes:

.. code-block:: bash

    cp -p senna/hash/suffix.lst suffix.lst

The tagger can then be trained as follows:

.. code-block:: bash

    bin/dl-ner.py ner.dnn -t train+dev.iob \
          --vocab vocab.txt --vectors vectors.txt \
          --caps --suffix --suffixes suffix.lst --gazetteer eng.list \
          -e 40 --variant senna \
          -l 0.01 -w 5 -n 300 -v

The benchmark can be run as:

.. code-block:: bash

    bin/dl-ner.py ner.dnn < testb.iob > testb.out.iob

The results I achieved are::

    processed 46435 tokens with 5648 phrases; found: 5640 phrases; correct: 5031.
    accuracy:  97.62%; precision:  89.20%; recall:  89.08%; FB1:  89.14
              LOC: precision:  93.30%; recall:  91.01%; FB1:  92.14
             MISC: precision:  78.24%; recall:  77.35%; FB1:  77.79
              ORG: precision:  84.59%; recall:  87.24%; FB1:  85.89
              PER: precision:  94.71%; recall:  94.06%; FB1:  94.38

Writing Extensions
==================

You can modify or extend the code just by adding them to the directory ``deepnl``.
To compile the extension, use the same build process, but you will also need to have Cython_ installed.
The compiler will issue warnings about NumPy of the type:

   /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
 #warning "Using deprecated NumPy API, disable it by "

Simply disregard them, since currently there is no way to fix them, until the maintainers of Cython_ will decide to upgrade it to use the latest API.

Credits
=======

Erick Fonseca developed ``nlpnet``, a similar library, available at:
https://github.com/erickrf/nlpnet, which provided inspiration for ``deepnl``.

References
==========

.. [Attardi] Giuseppe Attardi. 2015. DeepNL: a Deep Learning NLP
	     pipeline. Workshop on Vector Space Modeling for NLP, NAACL 2015,
	     Denver, Colorado (June 5, 2015).

.. [Collobert11] Ronan Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu and P. Kuksa.
   Natural Language Processing (Almost) from Scratch. *Journal of Machine
   Learning Research*, 12:2493-2537, 2011.

.. [Lebret14]  Rémi Lebret and Ronan  Collobert. 2014. Word Embeddings through Hellinger PCA. *EACL 2014*: 482.

.. _numpy: http://www.numpy.org
.. _Eigen: http://eigen.tuxfamily.org/
.. _Cython: http://cython.org
.. _SENNA: http://ronan.collobert.com/senna/
