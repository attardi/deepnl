===============================================================
``nlpnet`` --- Natural Language Processing with neural networks
===============================================================

``nlpnet`` is a Python library for Natural Language Processing tasks based on neural networks. 

It performs part-of-speech tagging, Named Entity tagging and semantic role
labeling.

This branch is rewrite by Giuseppe Attardi (http://www.di.unipi.it/~attardi)
to make it consistent with the architecture of SENNA_:

.. _SENNA: http://ronan.collobert.com/senna/

The original version by Erick Fonseca is available at: https://github.com/erickrf/nlpnet.

Dependencies
------------

``nlpnet`` requires numpy_.

NLTK_ is also needed to perform Portuguese tokenization and sentence splitting
when used as a library (see `Library usage`_).

The POS and NER command line taggers expect instead properly tokenized input.

Cython_ is required in development for generating C extensions that run faster. You probably won't need it, since the generated ``.c`` file is already provided with `nlpnet`, but you will need a C compiler.

.. _numpy: http://www.numpy.org
.. _Cython: http://cython.org
.. _NLTK: http://www.nltk.org

Basic usage
-----------

``nlpnet`` can be used both as a Python library or through command line scripts. Both usages are explained below.

Library usage
~~~~~~~~~~~~~

You can use ``nlpnet`` as a library in Python code as follows:

.. code-block:: python

    >>> import nlpnet
    >>> nlpnet.set_data_dir('/path/to/nlpnet-data/')
    >>> tagger = nlpnet.POSTagger()
    >>> tagger.tag('O rato roeu a roupa do rei de Roma.')
    [[(u'O', u'ART'), (u'rato', u'N'), (u'roeu', u'V'), (u'a', u'ART'), (u'roupa', u'N'), (u'do', u'PREP+ART'), (u'rei', u'N'), (u'de', u'PREP'), (u'Roma', u'NPROP'), (u'.', 'PU')]]

In the example above, the call to ``set_data_dir`` indicates where the data directory is located. This location must be given whenever ``nlpnet`` is imported. 

Calling a tagger is pretty straightforward. The provided taggers are:
``POSTagger``, ``NERTagger`` and ``SRLTagger``, all having a method ``tag`` which receives strings with text to be tagged. The tagger splits the text into sentences and then tokenizes each one (hence the return of the POSTagger is a list of lists).

The output of the ``NERTagger`` is in ``IOB`` notation.

The output of the ``SRLTagger`` is slightly more complicated:

    >>> tagger = nlpnet.SRLTagger()
    >>> tagger.tag(u'O rato roeu a roupa do rei de Roma.')
    [<nlpnet.taggers.SRLAnnotatedSentence at 0x84020f0>]

Instead of a list of tuples, sentences are represented by instances of ``SRLAnnotatedSentence``. This class serves basically as a data holder, and has two attributes:

    >>> sent = tagger.tag(u'O rato roeu a roupa do rei de Roma.')[0]
    >>> sent.tokens
    [u'O', u'rato', u'roeu', u'a', u'roupa', u'do', u'rei', u'de', u'Roma', u'.']
    >>> sent.arg_structures
    [(u'roeu',
      {u'A0': [u'O', u'rato'],
       u'A1': [u'a', u'roupa', u'do', u'rei', u'de', u'Roma'],
       u'V': [u'roeu']})]

The ``arg_structures`` is a list containing all predicate-argument structures in the sentence. The only one in this example is for the verb `roeu`. It is represented by a tuple with the predicate and a dictionary mapping semantic role labels to the tokens that constitute the argument.

Note that the verb appears as the first member of the tuple and also as the content of label 'V' (which stands for verb). This is because some predicates are multiwords. In these cases, the "main" predicate word (usually the verb itself) appears in ``arg_structures[0]``, and all the words appear under the key 'V'.

Standalone scripts
~~~~~~~~~~~~~~~~~~

``nlpnet`` also provides scripts for tagging text, training new models and testing them. They are copied to the `scripts-<python-version>` subdirectory of your Python installation, which can be included in the system PATH variable. You can call them from command line and give some text input.
The scripts expect tokenized input, one token per line, with an empty
line to separate sentences.
When training, the token attributes are supplied in tsv format.

.. code-block:: bash

    $ nlpnet-tag.py pos /path/to/nlpnet-data/
    O
    rato
    roeu
    a
    roupa
    do
    rei
    de
    Roma
    .

    O	ART
    rato	N
    roeu	V
    a	ART
    roupa	N
    do	PREP+ART
    rei	N
    de	PREP
    Roma	NPROP
    .	PU

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

To learn more about training and testing new models, and other functionalities, refer to the documentation at http://nilc.icmc.usp.br/nlpnet

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

Assuming that the SENNA distribution is in directory ``senna``,
the embeddings from SENNA can be adapted for use with ``nlpnet`` with the command:

.. code-block:: bash

    bin/embeddingsAdapter.py -s senna/embeddings/english.tsv model/types-features-ner.npy model/word-dict.pickle

The gazetters from SENNA can be used to produce a single entity list as follows:

.. code-block:: bash

    iconv -f ISO-8859-1 -t UTF-8 < senna/hash/ner.loc.lst | awk '{printf "LOC\t%s\n", $$0}'  > model/eng.list
    iconv -f ISO-8859-1 -t UTF-8 < senna/hash/ner.misc.lst | awk '{printf "MISC\t%s\n", $$0}' >> model/eng.list
    iconv -f ISO-8859-1 -t UTF-8 < senna/hash/ner.org.lst | awk '{printf "ORG\t%s\n", $$0}' >> model/eng.list
    iconv -f ISO-8859-1 -t UTF-8 < senna/hash/ner.per.lst | awk '{printf "PER\t%s\n", $$0}' >> model/eng.list

You also need the list of suffixes:

.. code-block:: bash

    cp -p senna/hash/suffix.lst model/suffixes.txt

The tagger can then be trained as follows:

.. code-block:: bash

    bin/nlpnet-train.py ner --load_features --gazetteer \
         --data model --gold train+dev.iob -e 40 --variant senna \
         -l 0.0001 --lf 0.01 --lt 0.01 -w 5 -n 300 --caps --suffix -v

The benchmark can be run as:

.. code-block:: bash

    bin/nlpnet-tag.py ner model < testb.iob > testb.out.iob

The results I achieved are::

processed 46435 tokens with 5648 phrases; found: 5640 phrases; correct: 5031.
accuracy:  97.62%; precision:  89.20%; recall:  89.08%; FB1:  89.14
              LOC: precision:  93.30%; recall:  91.01%; FB1:  92.14
             MISC: precision:  78.24%; recall:  77.35%; FB1:  77.79
              ORG: precision:  84.59%; recall:  87.24%; FB1:  85.89
              PER: precision:  94.71%; recall:  94.06%; FB1:  94.38
