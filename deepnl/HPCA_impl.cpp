
#include "Python.h"

#include "numpy/ndarraytypes.h"
#include "numpy/ndarrayobject.h"

#include <math.h>
#include <string.h>		// strtok_r()
#include <strings.h>		// bzero()
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

using namespace std;

namespace hpca {

bool verbose = true;

typedef unordered_map<string, unsigned> Dict;

template<class T>
struct ring
{
  ring(unsigned size) :
    buffer(size),
    _start(0),
    _end(0)
  { }

  void clear() {
    _start = _end = 0;
  }

  struct iterator {
    ring& r;
    T* curr;

    iterator(ring& r, T* curr) :
      r(r),
      curr(curr)
    {}

    T& operator *() { return *curr; }

    iterator& operator++() {
      if (++curr == &*r.buffer.end())
	curr = &*r.buffer.begin();
      if (curr == r._end)
	curr = 0;		// signals end
      return *this;
    }

    //! Increment operator (postfix).
    iterator operator ++ (int) {
        iterator tmp = *this;
        ++*this;
        return tmp;
    }

    bool operator ==(const iterator& it) const {
      return curr == it.curr;
    }

    bool operator !=(const iterator& it) const {
      return curr != it.curr;
    }

  };
  
  void add(T x) {
    if (empty())
      _start = _end = &*buffer.begin();
    else if (_end == _start) {
      if (++_start == &*buffer.end())
	_start = &*buffer.begin();
    }
    *_end = x;
    if (++_end == &*buffer.end())
      _end = &*buffer.begin();
  }

  bool empty() { return _start == 0; }

  unsigned size() {
    return (empty())
      ? 0
      : (_start < _end) ? _end - _start : buffer.size() - (_start - _end);
  }

  iterator begin() { return iterator(*this, _start); }
  iterator end() { return iterator(*this, 0); }

  vector<T> buffer;
  T* _start; ///< virtual beginning
  T* _end;   ///< virtual end (one besides last element)
};

static int MAX_LINE_LEN = 1 << 14;
/**
 *   load list of words from file
 */
void load_list(char const* file, Dict& dict)
{
  char line[MAX_LINE_LEN];
  ifstream ifs(file);
  if (!ifs) {
    cerr << "No such file: " << file << endl;
    return;
  }
  while (ifs.getline(line, MAX_LINE_LEN))
    dict[line] = dict.size() - 1; // lhs is evaluated first
}

#define P(i, j) (*(p + (i) * top + (j)))

/**
 * Build cooccurrence matix @param p from corpus of sentences in file @param
 * corpus, using vocabulary from file @c vocabFile, of which the top @param
 * top words are used as context words.
 */
PyObject* cooccurrence_matrix(char* corpus, char* vocabFile, unsigned top,
			      unsigned window)
{
  // read vocabulary
  Dict vocab;
  load_list(vocabFile, vocab);
  
  unsigned words = vocab.size();

  // allocate numpy array
  const npy_intp dims[2] = {words, top};
  PyObject* npyp = PyArray_ZEROS(2, (npy_intp*)dims, NPY_FLOAT32, NPY_CORDER);
  //Py_INCREF(npyp);
  float* p = (float*)PyArray_DATA(npyp);

  // read sentences
  char sentence[MAX_LINE_LEN];
  ifstream ifs(corpus);
  if (!ifs) {
    cerr << "No such file: " << corpus << endl;
    return npyp;
  }
  ring<unsigned> context(window);
  
  int sentCount = 0;
  while (true) {
    if (!ifs.getline(sentence, MAX_LINE_LEN)) {
      if (ifs.rdstate() & ifstream::failbit) {	// too long line
	ifs.clear();
	ifs.ignore(numeric_limits<streamsize>::max(), '\n');
	if (ifs.rdstate())
	  break;
	if (verbose)
	  cerr << "\nLong line: " << sentCount << endl;
	sentCount++;
	continue;
      } else
	break;
    }
    context.clear();
    char* next = sentence;
    char* tok = strtok_r(0, " ", &next);
    // count how many times a context word w in D appears after a vocabulary
    // word T in V, in a window of context-size: C(T, w)
    // context words are the first top in the vocabulary.
# if 1//def CONTEXT_SIZE
    while ((tok = strtok_r(0, " ", &next))) {
      Dict::const_iterator w = vocab.find(tok);
      if (w == vocab.end()) {
	context.add(-1);	// OOV
	continue;
      }
      // w in T
      if (w->second < top) {	// w in D
	for (int T : context) {
	  if (T >= 0)
	    P(T, w->second)++;	// p[T, w] = n(w, T) = C(T, w)
	}
      }
      context.add(w->second);
    }
# else
    char* prev = strtok_r(0, " ", &next);
    while ((tok = strtok_r(0, " ", &next))) {
      Dict::const_iterator w = vocab.find(tok);
      if (w != vocab.end()) {
	Dict::const_iterator T = vocab.find(prev);
	if (T != vocab.end() && w->second < top)
	    P(T->second, w->second)++; // p[T, w] = n(w, T)
      }
      prev = tok;
    }
# endif
    sentCount++;
    if (verbose) {
      if (sentCount % 100000 == 0) {
	cerr << '+';
	cerr.flush();
      } else if (sentCount % 10000 == 0) {
	cerr << '.';
	cerr.flush();
      }
    }
  }
  if (verbose) {
    cerr << endl;
    cerr << "Sentences: " << sentCount << endl;
  }
  // normalize counts and apply sqrt()
  if (verbose)
    cerr << "Normalize frequencies" << endl;
  for (unsigned j = 0; j < top; j++) {
    float nT = 0;		// Sum_w C(T, w) != C(T) (includes OOV w)
    for (unsigned i = 0; i < words; i++)
      nT += P(i, j);		// p[i, j] sum by column
    if (nT == 0.0)		// better doing a single test here
      nT = 1;			// avoid zero division
    for (unsigned i = 0; i < words; i++)
      P(i, j) = sqrt(P(i, j) / nT); // p[i, j]
  }
  //Py_DECREF(npyp); DEBUG
  return npyp;
}

} // namespace hpca

// needed to get PyArray_DescrFromType to work
int dummy = _import_array();
