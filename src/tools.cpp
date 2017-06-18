#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
    cout<< "Invalid estimations or ground truth data" << endl;
    return rmse;
  }

  //accumulate squared residuals
  for (unsigned i = 0; i < estimations.size(); ++i) {
    VectorXd v = estimations[i] - ground_truth[i];
    rmse = rmse.array() + v.array()*v.array();
  }

  //calculate the mean
  rmse = rmse.array()/estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  return rmse;
}