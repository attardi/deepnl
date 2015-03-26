===============================================================
``deepnl`` --- Deep Learning for Natural Language Processing
===============================================================

``deepnl`` is a Python library for Natural Language Processing tasks based on
neural networks.

The library currently provides tools for performing part-of-speech tagging,
Named Entity tagging and semantic role labeling.

Dependencies
------------

``deepnl`` requires numpy_.

The POS and NER command line taggers expect instead properly tokenized input.

Cython_ is required in development for generating C++ extensions that run faster.
The generated ``.cpp`` files are already provided with ``deepnl``, but you will need a C++ compiler.

.. _numpy: http://www.numpy.org
.. _Cython: http://cython.org

Basic usage
-----------

``deepnl`` can be used both as a Python library or through command line scripts. Both usages are explained below.

Library usage
~~~~~~~~~~~~~

You can use ``deepnl`` as a library in Python code as follows, where
``filename`` is the name of the file containing the model produced through training:

.. code-block:: python

    >>> import deepnl
    >>> tagger = deepnl.Tagger.load(filename)
    >>> sent = 'The quick brown fox jumped over the lazy dog.'
    >>> tagger.tag_sentence(sent.split(), return_tokens=True)
    [[(u'The', u'DT'), (u'quick', u'JJ'), (u'brown', u'JJ'), (u'fox', u'NN'), (u'jumped', u'VBD'), (u'over', u'IN'), (u'the', u'DT'), (u'lazy', u'JJ'), (u'dog', u'NN'), (u'.', '.')]]

Calling a tagger is pretty straightforward. The provided taggers are:
``PosTagger``, ``NerTagger`` and ``SrlTagger``, all having a method ``tag`` which receives strings with text to be tagged. The tagger splits the text into sentences and then tokenizes each one (hence the return of the ``PosTagger`` is a list of lists).

The output of the ``NerTagger`` is in ``IOB`` notation.


Standalone scripts
~~~~~~~~~~~~~~~~~~

``deepnl`` also provides scripts for tagging text, training new models and testing them. They are copied to the `scripts-<python-version>` subdirectory of your Python installation, which can be included in the system PATH variable. You can call them from command line and give some text input.
The scripts expect tokenized input, one token per line, with an empty
line to separate sentences.
When training, the token attributes are supplied in tsv format.
Here is an example of POS tagging, using the model in file ``pos.dnn``:

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

Benchmarks
~~~~~~~~~~

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

Assuming that the SENNA distribution is in directory ``senna``, the embeddgins
and vocabulary from SENNA can be used:

.. code-block:: bash

   cp -p senna/embeddings/embeddings.txt vectors.txt
   cp -p senna/hash/words.txt vocab.txt

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

    bin/dl-ner.py ner.dnn -t train+dev \
          --vocab vocab.txt --vectors vectors.txt \
          --caps --suffix --suffixes suffix.lst --gazetteer eng.list \
          -e 40 --variant senna \
          -l 0.01 -w 5 -n 300 -v

The benchmark can be run as:

.. code-block:: bash

    bin/dl-ner.py model < testb.iob > testb.out.iob

The results I achieved are::

.. code-block:: bash
    processed 46435 tokens with 5648 phrases; found: 5640 phrases; correct: 5031.
    accuracy:  97.62%; precision:  89.20%; recall:  89.08%; FB1:  89.14
              LOC: precision:  93.30%; recall:  91.01%; FB1:  92.14
             MISC: precision:  78.24%; recall:  77.35%; FB1:  77.79
              ORG: precision:  84.59%; recall:  87.24%; FB1:  85.89
              PER: precision:  94.71%; recall:  94.06%; FB1:  94.38

Credits
~~~~~~~~~~

Erick Fonseca developed ``nlpnet``, a similar library, available at:
https://github.com/erickrf/nlpnet, which provided inspiration for ``deepnl``.
