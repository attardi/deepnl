
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
    packages = ["deepnl"],
    ext_modules = cythonize(
        extensions,
        language="c++",
        nthreads=4),
    scripts = glob.glob("bin/*.py"),
    license = "MIT",
    version = "1.2.2",
    author = "Giuseppe Attardi <attardi@di.unipi.it>",
    author_email = "attardi@di.unipi.it",
    url = "https://github.com/attardi/deepnl",
    classifiers = [
        "Topic :: Text Processing :: Linguistic"
    ],
    requires = ["numpy (>= 1.8)"],
    long_description = readme()
)
