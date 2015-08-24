
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy as np
import glob

def readme():
    with open('README.rst') as f:
        text = f.read()
    return text

extensions = [
    Extension('deepnl/words',
              sources=["deepnl/words.pyx", "deepnl/WordsTrainer.cpp"],
              include_dirs=[np.get_include(),
                            "/usr/include/eigen3"],
              language="c++",
              extra_compile_args=["-fopenmp"]),
    Extension('deepnl/hpca',
              sources=["deepnl/hpca.pyx", "deepnl/HPCA.cpp"],
              include_dirs=[np.get_include(),
                            "/usr/include/eigen3"],
              language="c++",
              extra_compile_args=["-std=c++11"],
              extra_link_args=["-fopenmp"]),
    Extension('deepnl/*',
              sources=['deepnl/*.pyx'],
              include_dirs=[np.get_include(),
                            "/usr/include/eigen3"],
              language="c++",
              extra_compile_args=["-fopenmp"]),
]

setup(
    name = "deepnl",

    description = "Deep Learning for NLP tasks",
    author = "Giuseppe Attardi <attardi@di.unipi.it>",
    author_email = "attardi@di.unipi.it",
    url = "https://github.com/attardi/deepnl",

    license = "GNU GPL",
    version = "1.3.7",

    platforms = "any",

    keywords = " Deep learning "
        " Neural network "
        " Natural language processing ",

    requires = ["numpy (>= 1.9)"],

    packages = ["deepnl"],

    ext_modules = cythonize(
        extensions,
        language="c++",
        nthreads=4),
    scripts = glob.glob("bin/*.py"),

    classifiers = [
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],

    long_description = readme()
)
