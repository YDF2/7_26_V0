#ifndef __IMU_HPP
#define __IMU_HPP

#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <eigen3/Eigen/Dense>
#include "sensor.hpp"
#include "model.hpp"

extern "C" {
#include "hipnuc_dec.h"
#include "command_handlers.h"
}

enum FallDirection
{
    FALL_NONE = 0,
    FALL_FORWARD = 1,
    FALL_BACKWARD = -1,
    FALL_LEFT = 2,
    FALL_RIGHT = -2
};

class Imu: public Sensor
{
public:
    struct imu_data
    {
        float pitch=0.0, roll=0.0, yaw=0.0;
        float ax=0.0, ay=0.0, az=0.0;
        float wx=0.0, wy=0.0, wz=0.0;
        int timestamp=0;
    };

    Imu();
    ~Imu();

    bool start();
    void stop();

    imu_data data() const
    {
        return imu_data_;
    }

    int fall_direction() const
    {
        return fall_direction_;
    }
private:
    bool open();
    void run();

    hipnuc_raw_t hipnuc_raw_;
    imu_data imu_data_;
    std::thread td_;

    std::atomic_int fall_direction_;

    Eigen::Vector2f pitch_range_;
    Eigen::Vector2f roll_range_;

    float init_dir_;

    int fd_;
    uint8_t recv_buf_[1024];
};

#endif
