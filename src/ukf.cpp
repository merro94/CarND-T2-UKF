#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <fstream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  //set state dimension
  n_x_ = 5;

  //set augmented dimension
  n_aug_ = 7;

  //define spreading parameter
  lambda_ = 3 - n_aug_;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // set vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  double w = 0.5 / (n_aug_ + lambda_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    weights_(i) = w;
  }

  NIS_radar_ = 0.;
  NIS_laser_ = 0.;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state x_ with the first measurement.
      * Create the covariance matrix.
    */
    // first measurement
    cout << "UKF: " << endl;
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;

    // first measurement
     float px;
     float py;
     float v;
     float yaw;
     float yawd;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rhodot = meas_package.raw_measurements_[2];

      px = rho * cos(phi);
      py = rho * sin(phi);
      v = 0.;
      yaw = 0.;
      yawd = 0.;

    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];
      v = 0.;
      yaw = 0.;
      yawd = 0.;
    }

    x_ << px, py, v, yaw, yawd;

    time_us_ = meas_package.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // compute the time elapsed between the current and previous measurements
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0; //dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
     * and output the NIS values
   */

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    if (use_radar_) {
      UpdateRadar(meas_package);
      cout << "NIS_radar = " << NIS_radar_ << "\n";
    }

  } else {
    // Laser updates
    if (use_laser_) {
      UpdateLidar(meas_package);
      cout << "NIS_laser = " << NIS_laser_ << "\n";
    }
  }

  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  //create augmented matrix with predicted sigma points
  // Xsig_pred_ is updated afterwards
  SigmaPointPrediction(delta_t);

  //create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);

  //predict state mean
  x = Xsig_pred_ * weights_;

  //predict state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    // angle normalization
    while (x_diff(3) >  M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.*M_PI;

    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }

  //update state
  x_ = x;
  P_ = P;
}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 2;

  //Sigma points already generated in Prediction step and stored in Xsig_pred_

  //create matrix with sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  //create vector for mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  //create matrix for predicted measurement covariance
  MatrixXd S = MatrixXd(n_z, n_z);

  //transform sigma points into measurement space
  for (int i = 0; i < n_aug_ * 2 + 1; i++) {
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    Zsig(0,i) = px;
    Zsig(1,i) = py;
  }

  //calculate mean predicted measurement
  z_pred = Zsig * weights_;

  //calculate measurement covariance matrix S
  S.fill(0);
  S(0,0) = std_laspx_ * std_laspx_;
  S(1,1) = std_laspy_ * std_laspy_;

  for (int i = 1; i < n_aug_ * 2 + 1; i++) {
    VectorXd diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (diff(1) >  M_PI) diff(1) -= 2.*M_PI;
    while (diff(1) < -M_PI) diff(1) += 2.*M_PI;

    S = S + weights_(i) * diff * diff.transpose();
  }

  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z(0) = meas_package.raw_measurements_[0];
  z(1) = meas_package.raw_measurements_[1];

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_aug_ * 2 + 1; i++) {

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3) >  M_PI) x_diff(3) -= 2.* M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.* M_PI;

    VectorXd z_diff = Zsig.col(i) - z_pred;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd z_res = z - z_pred;

  x_ = x_ + K * z_res;
  P_ = P_ - K * S * K.transpose();
  NIS_laser_ = z_res.transpose() * S.inverse() * z_res;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //Sigma points already generated in Prediction step and stored in Xsig_pred_

  //create matrix with sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  //create vector for mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  //create matrix for predicted measurement covariance
  MatrixXd S = MatrixXd(n_z, n_z);

  //transform sigma points into measurement space
  for (int i = 0; i < n_aug_ * 2 + 1; i++) {
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    double yawd = Xsig_pred_(4,i);
    Zsig(0,i) = sqrt(px*px + py*py);
    Zsig(1,i) = atan2(py, px);
    Zsig(2,i) = (px*cos(yaw)*v + py*sin(yaw)*v) / Zsig(0,i);
  }

  //calculate mean predicted measurement
  z_pred = Zsig * weights_;

  //calculate measurement covariance matrix S
  S.fill(0);
  S(0,0) = std_radr_ * std_radr_;
  S(1,1) = std_radphi_ * std_radphi_;
  S(2,2) = std_radrd_ * std_radrd_;

  for (int i = 1; i < n_aug_ * 2 + 1; i++) {
    VectorXd diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (diff(1) >  M_PI) diff(1) -= 2.*M_PI;
    while (diff(1) < -M_PI) diff(1) += 2.*M_PI;

    S = S + weights_(i) * diff * diff.transpose();
  }

  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z(0) = meas_package.raw_measurements_[0];
  z(1) = meas_package.raw_measurements_[1];
  z(2) = meas_package.raw_measurements_[2];

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_aug_ * 2 + 1; i++) {

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3) >  M_PI) x_diff(3) -= 2.* M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.* M_PI;

    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1) >  M_PI) z_diff(1) -= 2.* M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.* M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd z_res = z - z_pred;
  //angle normalization
  while (z_res(1) >  M_PI) z_res(1) -= 2.*M_PI;
  while (z_res(1) < -M_PI) z_res(1) += 2.*M_PI;

  x_ = x_ + K * z_res;
  P_ = P_ - K * S * K.transpose();
  NIS_radar_ = z_res.transpose() * S.inverse() * z_res;

}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  //create augmented covariance matrix
  P_aug.fill(0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  double w = sqrt(n_aug_ + lambda_);

  for (int i=0; i < n_aug_ ; i++) {
    Xsig_aug.col(i+1)        = x_aug + w * A.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - w * A.col(i);
  }

  //write result
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(double delta_t) {

  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  AugmentedSigmaPoints(&Xsig_aug);

  double delta_t_2 = 0.5 * delta_t * delta_t;

  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * ( sin (yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw + yawd * delta_t) );
    }
    else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + nu_a * delta_t_2 * cos(yaw);
    py_p = py_p + nu_a * delta_t_2 * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + nu_yawdd * delta_t_2;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

}
