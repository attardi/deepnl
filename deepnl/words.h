
// Eigen library
#include <Eigen/Dense>

typedef Eigen::VectorXd	Vector;
typedef Eigen::MatrixXd	Matrix;

struct Parameters
{
  Matrix wIn;		// input weights
  Vector bIn;		// input bias
  Matrix wOut;		// output weights
  Vector bOut;		// output bias

  Parameters() { }

  Parameters(unsigned numInput, unsigned numHidden, unsigned numOutput) :
    wIn(numInput, numHidden),
    bIn(numHidden),
    wOut(numHidden, numOutput),
    bOut(numOutput)
  {
    clear();
  }

  virtual void clear() {
    wIn.setZero();
    bIn.setZero();
    wOut.setZero();
    bOut.setZero();
  }

};

class Network
{
  Vector	hidden, output;	///< hidden and output variables
  unsigned	numInput;	///< number of input values
  unsigned	numHidden;	///< number of hidden variables
  unsigned	numOutput;	///< number of output variables
  Parameters*	p;		///< parameters

  void run(const Vector& input) {
    hidden.noalias() = input.transpose() * p->wIn + p->bIn.transpose();
    tanh(hidden);		// first layer
    output.noalias() = hidden.transpose() * p->wOut + p->bOut.transpose();
  }
};

class Trainer
{
public:

  void init(PyObject* nn);

  void train_pair(double* pos_input_values_0, double* negative_token_0);
};

