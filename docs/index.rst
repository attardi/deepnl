.. py:module:: nlpnet

==================================================================
:mod:`nlpnet` --- Natural Language Processing with neural networks
==================================================================

:mod:`nlpnet` is a Python library for Natural Language Processing tasks based on neural networks. 
Currently, it performs part-of-speech tagging and semantic role labeling. It may be used as a Python
library or through its standalone scripts. Most of the architecture is language independent, 
but some functions were especially tailored for working with Portuguese.

This system was inspired by SENNA_, but has some conceptual and practical differences. 
If you use :mod:`nlpnet`, please cite one or both of the articles below, according to your needs (POS or
SRL):

.. _SENNA: http://ronan.collobert.com/senna/

* Fonseca, E. R. and Rosa, J.L.G. *A Two-Step Convolutional Neural Network Approach for Semantic
  Role Labeling*. Proceedings of the 2013 International Joint Conference on Neural Networks, 2013.
  p. 2955-2961 [`PDF <http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6707118>`_]

* Fonseca, E. R. and Rosa, J.L.G. *Mac-Morpho Revisited: Towards Robust Part-of-Speech Tagging*. 
  Proceedings of the 9th Brazilian Symposium in Information and Human Language Technology, 2013. p.  
  98-107 [`PDF <http://aclweb.org/anthology//W/W13/W13-4811.pdf>`_]

Contents
--------

.. toctree::
    :maxdepth: 2

    intro
    scripts
    utils
    network
    
