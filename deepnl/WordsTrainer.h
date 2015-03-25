
#include <iostream>		// DEBUG

// Eigen library
#include <Eigen/Dense>

typedef Eigen::VectorXd	Vector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>	Matrix;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix1i;

using Eigen::Map;

inline void hardtanh(Vector const& xc)
{
  Vector& x = const_cast<Vector&>(xc);
  for (unsigned i = 0; i < x.size(); i++)
    if (x(i) < -1.0)
      x(i) = -1.0;
    else if (x(i) > 1)
      x(i) = 1.0;
}

/**
 * Converse of hardtanh: computes derivative of input, given the output, i.e,
 * given y = hardtanh(x), compute hardtanh'(x).
 */
inline Vector& hardtanhe(Vector const& cy)
{
  Vector& y = const_cast<Vector&>(cy);
  for (unsigned i = 0; i < y.size(); i++)
    if (y(i) == -1.0 || y(i) == 1.0)
      y(i) = 0.0;
    else
      y(i) = 1.0;
  return y;
}

struct Parameters
{
  Map<Matrix> hidden_weights;	// input weights
  Map<Vector> hidden_bias;	// input bias
  Map<Matrix> output_weights;	// output weights
  Map<Vector> output_bias;	// output bias

  Parameters(int numInput, int numHidden, int numOutput,
	     double* hidden_weights, double* hidden_bias,
	     double* output_weights, double* output_bias) :
    hidden_weights(hidden_weights, numHidden, numInput),
    hidden_bias(hidden_bias, numHidden),
    output_weights(output_weights, numOutput, numHidden),
    output_bias(output_bias, numOutput)
  { }

};

struct Network : public Parameters
{
  Network(int numInput, int numHidden, int numOutput,
	  double* hidden_weights, double* hidden_bias,
	  double* output_weights, double* output_bias) :
    Parameters(numInput, numHidden, numOutput,
	       hidden_weights, hidden_bias,
	       output_weights, output_bias)
  { }

  void run(const Vector& input, Vector const& hidden, Vector const& output) {
    // We must pass const and then cast it away, a hack according to:
    // http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
    const_cast<Vector&>(hidden).noalias() = hidden_weights * input + hidden_bias;
    hardtanh(hidden);		// first layer
    const_cast<Vector&>(output).noalias() = output_weights * hidden + output_bias;
  }
};

struct LmGradients : public Parameters
{
  Map<Vector> input_pos;	// positive input variables
  Map<Vector> input_neg;	// negative input variables

  LmGradients(int numInput, int numHidden, int numOutput,
	      double* hiddenWeights, double* hiddenBias,
	      double* outpuWeights, double* outputBias,
	      double* input_pos, double* input_neg) :
    Parameters(numInput, numHidden, numOutput,
	       hiddenWeights, hiddenBias,
	       outpuWeights, outputBias),
    input_pos(input_pos, numInput),
    input_neg(input_neg, numInput)
  { }
};

class WordsTrainer
{
public:

  WordsTrainer(int numInput, int numHidden, int numOutput,
	       double* hidden_weights, double* hidden_bias,
	       double* output_weights, double* output_bias,
	       double* input_pos, double* input_neg,
	       double* grads_hidden_weights, double* grads_hidden_bias,
	       double* grads_output_weights, double* grads_output_bias,
	       double* grads_input_pos, double* grads_input_neg,
	       int* example, int window_size,
	       double* table, int table_rows, int table_cols) :
    nn(numInput, numHidden, numOutput,
       hidden_weights, hidden_bias,
       output_weights, output_bias),
    input_pos(input_pos, numInput),
    input_neg(input_neg, numInput),
    hidden_pos(numHidden),
    hidden_neg(numHidden),
    output_pos(numOutput),
    output_neg(numOutput),
    grads(numInput, numHidden, numOutput,
	  grads_hidden_weights, grads_hidden_bias,
	  grads_output_weights, grads_output_bias,
	  grads_input_pos, grads_input_neg),
    example(example, window_size, 1),
    window_size(window_size),
    table(table, table_rows, table_cols)
  { }

  // input from: input_pos, input_neg, output to: output_pos, output_neg
  double train_pair();

  void	update_embeddings(double LR_0,
			  int token_pos,
			  int token_neg);

  Network	nn;
  Map<Vector>	input_pos, input_neg; // shared with python
  Vector	hidden_pos, hidden_neg;
  Vector	output_pos, output_neg;
  LmGradients	grads;
  Map<Matrix1i>	example;
  int		window_size;
  Map<Matrix>	table;		// word vectors
};
