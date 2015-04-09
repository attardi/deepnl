
CYTHON_FILES = $(wildcard deepnl/*.pyx)
EXT_FILES = $(CYTHON_FILES:.pyx=.cpp)

all: #$(EXT_FILES)
	python setup.py build

%.cpp: %.pyx
	cython $< --cplus
