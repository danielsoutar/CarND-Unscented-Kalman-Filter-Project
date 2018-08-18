#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;


Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  int STATE_SIZE = 4;
  VectorXd RMSE(STATE_SIZE);
  VectorXd delta(STATE_SIZE);

  int n = estimations.size();

  if(n == 0 || n != ground_truth.size()) {
    cout << "CalculateRMSE() - Error - invalid vector sizes\n";
    return RMSE;
  }

  RMSE.fill(0);

  for (int i = 0; i < n; ++i) {
    delta = estimations[i] - ground_truth[i];
    delta = delta.array() * delta.array();
    RMSE += delta;
  }

  RMSE /= n;

  RMSE = RMSE.array().sqrt();

  return RMSE;

}

float Tools::CalculateNIS(const VectorXd z_pred, const VectorXd z_meas, 
                          const MatrixXd S) {
  VectorXd difference = z_meas - z_pred;
	float epsilon = difference.transpose() * S.inverse() * difference;
  return epsilon;
}