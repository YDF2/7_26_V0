#include "ball_ekf.hpp"
#include <cmath>

BallEKF::BallEKF()
{
    H_ << 1, 0, 0, 0,
          0, 1, 0, 0;
    init();
}

void BallEKF::init()
{
    status_ = UNINIT;
    coast_cnt_ = 0;
    x_.setZero();
    P_ << 0.01f, 0,     0,     0,
          0,     0.01f, 0,     0,
          0,     0,     0.001f, 0,
          0,     0,     0,     0.001f;
    F_.setIdentity();
    Q_.setZero();
    R_.setZero();
    sigma_meas_ = kSigmaBase;
    prev_pitch_rate_ = 0;
    prev_roll_rate_ = 0;
}

void BallEKF::predict(float dt)
{
    if (status_ == UNINIT)
        return;

    // Build F with current dt
    F_ << 1, 0, dt, 0,
          0, 1, 0,  dt,
          0, 0, 1,  0,
          0, 0, 0,  1;

    // State prediction
    x_ = F_ * x_;

    // Process noise Q (CWNA)
    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt2 * dt2;
    Q_ << kQC * dt4 / 4, 0,             kQC * dt3 / 2, 0,
          0,             kQC * dt4 / 4, 0,             kQC * dt3 / 2,
          kQC * dt3 / 2, 0,             kQC * dt2,     0,
          0,             kQC * dt3 / 2, 0,             kQC * dt2;

    // Covariance prediction
    P_ = F_ * P_ * F_.transpose() + Q_;

    if (status_ == COASTING)
        coast_cnt_++;
}

void BallEKF::update(float alpha, float beta)
{
    if (status_ == UNINIT)
    {
        reinit(alpha, beta);
        return;
    }

    // Measurement
    Eigen::Vector2f z(alpha, beta);

    // Innovation
    Eigen::Vector2f y = z - H_ * x_;

    // Innovation covariance
    Eigen::Matrix2f S = H_ * P_ * H_.transpose() + R_;

    // Mahalanobis gate — only in COASTING
    if (status_ == COASTING)
    {
        float d2 = y.transpose() * S.inverse() * y;
        if (d2 >= kGateThresh)
        {
            coast_cnt_++;  // rejected, stay in COASTING
            return;
        }
    }

    // Kalman gain
    Eigen::Matrix<float, 4, 2> K = P_ * H_.transpose() * S.inverse();

    // Update state
    x_ += K * y;

    // Update covariance (Joseph form for numerical stability)
    Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f IKH = I - K * H_;
    P_ = IKH * P_ * IKH.transpose() + K * R_ * K.transpose();

    status_ = TRACKING;
    coast_cnt_ = 0;
}

void BallEKF::reinit(float alpha, float beta)
{
    x_ << alpha, beta, 0, 0;
    P_ << 0.01f, 0,     0,     0,
          0,     0.01f, 0,     0,
          0,     0,     0.001f, 0,
          0,     0,     0,     0.001f;
    status_ = TRACKING;
    coast_cnt_ = 0;
}

void BallEKF::set_imu_shake(float pitch_rate, float roll_rate)
{
    // Low-pass filter the shake rates to avoid R jumping
    const float alpha = 0.7f;
    prev_pitch_rate_ = alpha * prev_pitch_rate_ + (1 - alpha) * pitch_rate;
    prev_roll_rate_  = alpha * prev_roll_rate_  + (1 - alpha) * roll_rate;

    float shake = prev_pitch_rate_ + prev_roll_rate_;
    float var = kSigmaBase * kSigmaBase + kShakeScale * shake;
    R_ << var, 0,
          0,   var;
    sigma_meas_ = std::sqrt(var);

    // Coasting: exceed max → reset
    if (status_ == COASTING && coast_cnt_ > kCoastMax)
    {
        status_ = UNINIT;
        coast_cnt_ = 0;
        x_.setZero();
    }
}
