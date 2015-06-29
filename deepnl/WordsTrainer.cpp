
#include "WordsTrainer.h"

double minError = 1.e-5;

double WordsTrainer::train_pair()
{
  nn.run(input_pos, hidden_pos, output_pos);
  nn.run(input_neg, hidden_neg, output_neg);

  // hinge loss
  double score_pos = output_pos[0];
  double score_neg = output_neg[0];
  double error = std::max(0.0, 1.0 - score_pos + score_neg);
  if (error < minError)
    return error;

  // (output_size) x (hidden_size) = (output_size, hidden_size)
  // (1) x (hidden_size) = (1, hidden_size)
  grads.output_weights.row(0) += hidden_pos - hidden_neg;

  // layer 2
  // (hidden_size) = (hidden_size) * (1, hidden_size)
  // hidden_pos = hardtanhe(hidden_pos) * nn.output_weights[0]
  hidden_pos = hardtanhe(hidden_pos).transpose().cwiseProduct(nn.output_weights.row(0));

  // hidden_neg = hardtanhe(nn.hidden_values) * (- nn.output_weights[0])
  hidden_neg = hardtanhe(hidden_neg).transpose().cwiseProduct(- nn.output_weights.row(0));

  // (hidden_size, input_size) = (hidden_size) x (input_size)
  grads.hidden_weights += hidden_pos * input_pos.transpose() +
    hidden_neg * input_neg.transpose();
  grads.hidden_bias += hidden_pos + hidden_neg;

  // input gradients
  // These are not accumulated, since their update is immediate.
  // (input_size) = (1, hidden_size) x (hidden_size, input_size)
  grads.input_pos = hidden_pos.transpose() * nn.hidden_weights;
  grads.input_neg = hidden_neg.transpose() * nn.hidden_weights;

  return error;
}

void WordsTrainer::update_embeddings(double LR_0, int token_pos, int token_neg)
{
  int middle = window_size/2;
  int start = 0;
  for (int i = 0; i < window_size; i++) {
    int end = start + table.cols();
    if (i == middle) {
      // this is the middle position.
      // apply negative and positive deltas to different tokens
      table.row(token_pos) += LR_0 * grads.input_pos.segment(start, end);
      table.row(token_neg) += LR_0 * grads.input_neg.segment(start, end);
    } else {
      // this is not the middle position. both deltas apply.
      int token = example(i, 0);
      table.row(token) += LR_0 * grads.input_pos.segment(start, end)
	+ LR_0 * grads.input_neg.segment(start, end);
    }	
    start = end;
  }
}
