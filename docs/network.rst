..  _network:

========
Networks
========

This document describes the neural networks used by :mod:`nlpnet`. They are written in Cython and use ``numpy`` a lot in order to attain very fast performance.

Note that these classes are somewhat low-level and don't deal with words and sentences explicitly. Instead, only vectorial representations are used. This approach is explained in the papers linked in the root page of this documentation.

.. :module:: nlpnet.network

Module :mod:`nlpnet.network`
============================

This module includes the actual neural networks. There are two classes of networks currently used: :class:`nlpnet.network.Network` for POS and :class:`nlpnet.network.ConvolutionalNetwork` for SRL.

.. :class::`nlpnet.network.Network`

Class :class:`nlpnet.network.Network`
-------------------------------------

.. autoclass:: nlpnet.network.Network
    :members: create_new, description, run, tag_sentence, train, save, load_from_file



.. :class::`nlpnet.network.ConvolutionalNetwork`

Class :class:`nlpnet.network.ConvolutionalNetwork`
--------------------------------------------------

.. autoclass:: nlpnet.network.ConvolutionalNetwork
    :members: create_new, description, run, tag_sentence, train, save, load_from_file
