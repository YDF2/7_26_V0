#ifndef __BALL_EKF_HPP
#define __BALL_EKF_HPP

#include <eigen3/Eigen/Dense>

class BallEKF
{
public:
    struct State
    {
        float alpha, beta;
        float valpha, vbeta;
    };

    BallEKF();

    void init();
    void predict(float dt);
    void update(float alpha, float beta);
    void reinit(float alpha, float beta);
    void set_imu_shake(float pitch_rate, float roll_rate);

    State state() const { return {x_(0), x_(1), x_(2), x_(3)}; }
    bool is_tracking() const { return status_ != UNINIT; }
    int coast_count() const { return status_ == COASTING ? coast_cnt_ : (status_ == TRACKING ? 0 : -1); }
    bool is_coasting() const { return status_ == COASTING; }

    static constexpr int kCoastMax = 4;           // 200ms @ 50ms period

private:
    enum { UNINIT, TRACKING, COASTING } status_;
    int coast_cnt_;

    Eigen::Vector4f x_;                            // [alpha, beta, valpha, vbeta]
    Eigen::Matrix4f P_;
    Eigen::Matrix4f F_;
    Eigen::Matrix4f Q_;
    Eigen::Matrix<float, 2, 4> H_;
    Eigen::Matrix2f R_;

    float sigma_meas_;
    float prev_pitch_rate_;
    float prev_roll_rate_;

    static constexpr float kSigmaBase = 0.002f;
    static constexpr float kShakeScale = 7e-7f;  // ZED Mini ~90° HFOV
    static constexpr float kQC = 0.005f;
    static constexpr float kGateThresh = 9.0f;
};

#endif
