#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

int STATE_SIZE = 5;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(STATE_SIZE);
  x_.fill(0);

  // initial covariance matrix
  P_ = MatrixXd::Identity(STATE_SIZE, STATE_SIZE);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
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
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  // This is our internal state - we'll update this to pass back to main.cpp in the form of a 4D vector containing (px, py, vx, vy).
  n_x_ = STATE_SIZE;

  // Recommended from class
  lambda_ = 3 - n_x_;

  // Add in the process noises in augmented state vector
  n_aug_ = 7;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  
  time_us_ = 0;

  // Set weights vector once
  weights_ = VectorXd(2*n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  weights_.tail(2*n_aug_) = VectorXd::Constant(2*n_aug_, 1 / (2 * (lambda_ + n_aug_)));
}

UKF::~UKF() {}

void UKF::GenerateSigmaPoints(MatrixXd& Xsig_aug, VectorXd& x_aug, MatrixXd& P_aug) {
  x_aug << x_, 0, 0;
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.block(n_x_, n_x_, 2, 2) << std_a_ * std_a_, 0, 
                                   0, std_yawdd_ * std_yawdd_;

  MatrixXd A = P_aug.llt().matrixL();

  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    VectorXd spread = sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(1 + i) = x_aug + spread;
    Xsig_aug.col(1 + n_aug_ + i) = x_aug - spread;
  }
} 

void UKF::PredictSigmaPoints(MatrixXd& Xsig_aug, float delta_t) {
  float delta_t_sq = delta_t * delta_t;

  VectorXd noiseVec(n_x_);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd sigma_point = Xsig_aug.col(i);

    float v = sigma_point(2);
    float psi = sigma_point(3);
    float psi_dot = sigma_point(4);
    float nu_a = sigma_point(5);
    float nu_psi = sigma_point(6);

    noiseVec << 0.5 * delta_t_sq * cos(psi) * nu_a,
                0.5 * delta_t_sq * sin(psi) * nu_a,
                delta_t * nu_a,
                0.5 * delta_t_sq * nu_psi,
                delta_t * nu_psi;

    if(psi_dot == 0) {
      Xsig_pred_.col(i) << (v * cos(psi) * delta_t),
                           (v * sin(psi) * delta_t),
                           0,
                           0,
                           0;
    }
    else {
      float frac = v / psi_dot;
      Xsig_pred_.col(i) << frac * (sin(psi + psi_dot*delta_t) - sin(psi)),
                           frac * (-cos(psi + psi_dot*delta_t) + cos(psi)),
                           0,
                           psi_dot * delta_t,
                           0;
    }
    Xsig_pred_.col(i) += noiseVec + sigma_point.head(n_x_);
  }
}

void UKF::PredictMeanAndCovariance() {
  x_.fill(0);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  P_.fill(0);

  for(int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd delta = Xsig_pred_.col(i) - x_;

    while(delta(3) > M_PI) delta(3) -= 2 * M_PI;
    while(delta(3) < -M_PI) delta(3) += 2 * M_PI;
    P_ += weights_(i) * delta * delta.transpose();
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  VectorXd x_aug(n_aug_);
  MatrixXd P_aug(n_aug_, n_aug_);
  MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);
  P_aug.fill(0);

  GenerateSigmaPoints(Xsig_aug, x_aug, P_aug);

  PredictSigmaPoints(Xsig_aug, delta_t);

  PredictMeanAndCovariance();
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if(!is_initialized_) {
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      float rho = meas_package.raw_measurements_[0]; // Distance, presumably in metres
      float phi = meas_package.raw_measurements_[1]; // Angle, in radians
      float rho_dot = meas_package.raw_measurements_[2];

      float px = std::cos(phi) * rho;
      float py = std::sin(phi) * rho;

      // Need to change to just 'v'
      float v = rho_dot;

      // Need to choose sensible default value of psi and psi dot
      // For psi dot, assume 0 (object is initially moving in straight line).
      // For psi itself, set to phi.
      float psi = phi;
      float psi_dot = 0;

      x_ << px, py, v, psi, psi_dot;
    }
    else if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // No information about velocity - just set the position, everything else to 0, and pray.
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  // Calculate delta_t and update previous time measurement for next iteration
  float delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(delta_t);

  if(meas_package.sensor_type_ == MeasurementPackage::RADAR) { // radar
    UpdateRadar(meas_package);
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::LASER) { // lidar
    UpdateLidar(meas_package);
  }
}

void UKF::Update(MatrixXd& Zsig, VectorXd& z_pred, VectorXd& z_meas, MatrixXd& S) {
  MatrixXd Tc(n_x_, z_pred.size());
  Tc.fill(0);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    Tc += weights_(i) * (Xsig_pred_.col(i) - x_) * (Zsig.col(i) - z_pred).transpose();
  }

  MatrixXd K = Tc * S.inverse();

  x_ += K * (z_meas - z_pred);

  P_ -= K * S * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Number of dimensions for lidar measurements.
  int n_z = 2;

  MatrixXd Zsig(n_z, 2 * n_aug_ + 1);

  VectorXd z_pred(n_z);
  z_pred.fill(0);

  MatrixXd S(n_z, n_z);
  S.fill(0);

  // Calculate mean of measurement
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x = Xsig_pred_.col(i);

    float px = x(0);
    float py = x(1);

    Zsig.col(i) << px, py;
    z_pred += weights_(i) * Zsig.col(i);
  }

  MatrixXd R(n_z, n_z);
  R << std_laspx_*std_laspx_, 0,
       0, std_laspy_*std_laspy_;

  // Calculate covariance of measurement
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd difference = (Zsig.col(i) - z_pred);
    S += weights_(i) * difference * difference.transpose();
  }

  S += R;

  VectorXd z_meas(n_z);
  z_meas << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1);

  Update(Zsig, z_pred, z_meas, S);

  // std::cout << "NIS: " << tool.CalculateNIS(z_pred, z_meas, S) << "\n";
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Number of dimensions for radar measurements.
  int n_z = 3;

  MatrixXd Zsig(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0);

  // Calculate mean of measurement
  for(int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x = Xsig_pred_.col(i);

    float px = x(0);
    float py = x(1);
    float v = x(2);
    float psi = x(3);

    float rho = sqrt(px*px + py*py);
    float phi = atan2(py, px);
    float rho_dot = (px * cos(psi) * v + py * sin(psi) * v) / rho;

    Zsig.col(i) << rho, phi, rho_dot;
    z_pred += weights_(i) * Zsig.col(i);

    while(z_pred(1) > M_PI) z_pred(1) -= 2 * M_PI;
    while(z_pred(1) < -M_PI) z_pred(1) += 2 * M_PI;
  }

  MatrixXd R(n_z, n_z);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;

  // Calculate covariance of measurement
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd difference = (Zsig.col(i) - z_pred);
    S += weights_(i) * difference * difference.transpose();
  }

  S += R;

  VectorXd z_meas(n_z);
  z_meas << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), meas_package.raw_measurements_(2);

  Update(Zsig, z_pred, z_meas, S);

  // std::cout << "NIS: " << tool.CalculateNIS(z_pred, z_meas, S) << "\n";
}





































