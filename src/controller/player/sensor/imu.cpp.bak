#include "imu.hpp"
#include "configuration.hpp"
#include "math/math.hpp"
#include "core/worldmodel.hpp"
#include "logger.hpp"
#include "core/clock.hpp"

using namespace std;
using namespace robot_math;
using namespace Eigen;

const float g_ = 9.8;

Imu::Imu(): Sensor("imu"), fd_(-1)
{
    Vector2f range = CONF->get_config_vector<float, 2>("not_fall_range.pitch");
    pitch_range_.x() = range[0];
    pitch_range_.y() = range[1];
    range = CONF->get_config_vector<float, 2>("not_fall_range.roll");
    roll_range_.x() = range[0];
    roll_range_.y() = range[1];
    fall_direction_ = FALL_NONE;
    init_dir_ = 0.0;
    memset(&hipnuc_raw_, 0, sizeof(hipnuc_raw_));
    memset(recv_buf_, 0, sizeof(recv_buf_));
}

bool Imu::open() 
{
    string dev_name = CONF->get_config_value<string>("hardware.imu.dev_name");
    int baudrate = CONF->get_config_value<int>("hardware.imu.baudrate");
    
    fd_ = read_start(const_cast<char*>(dev_name.c_str()), baudrate, recv_buf_);
    if(fd_ > 0) 
        return true;
    else 
    {
        LOG(LOG_WARN) << "imu: failed to open " << dev_name << endll;
        return false;
    }
}

void Imu::run()
{
    // Skip first 20 data packets as they may be invalid
    for(int i = 1; i <= 20 && is_alive_; i++)
    {
        imu_data_.timestamp = CLOCK->get_timestamp();
        read_loop(const_cast<char*>(CONF->get_config_value<string>("hardware.imu.dev_name").c_str()), 
                  CONF->get_config_value<int>("hardware.imu.baudrate"), 
                  fd_, recv_buf_, &hipnuc_raw_);
        usleep(1000);
    }
    
    while(is_alive_)
    {
        read_loop(const_cast<char*>(CONF->get_config_value<string>("hardware.imu.dev_name").c_str()), 
                  CONF->get_config_value<int>("hardware.imu.baudrate"), 
                  fd_, recv_buf_, &hipnuc_raw_);
        
        // Update IMU data from hi91 packet if available
        if(hipnuc_raw_.hi91.tag == 0x91)
        {
            imu_data_.pitch = hipnuc_raw_.hi91.pitch;
            imu_data_.roll = hipnuc_raw_.hi91.roll;
            imu_data_.yaw = hipnuc_raw_.hi91.yaw;
            
            // Acceleration data (convert from g to m/s²)
            imu_data_.ax = hipnuc_raw_.hi91.acc[0] * g_;
            imu_data_.ay = hipnuc_raw_.hi91.acc[1] * g_;
            imu_data_.az = hipnuc_raw_.hi91.acc[2] * g_;
            
            // Gyroscope data (deg/s)
            imu_data_.wx = hipnuc_raw_.hi91.gyr[0];
            imu_data_.wy = hipnuc_raw_.hi91.gyr[1];
            imu_data_.wz = hipnuc_raw_.hi91.gyr[2];
        }
        
        // Check fall direction
        if(imu_data_.pitch < pitch_range_.x()) 
            fall_direction_ = FALL_BACKWARD;
        else if(imu_data_.pitch > pitch_range_.y()) 
            fall_direction_ = FALL_FORWARD;
        else if(imu_data_.roll < roll_range_.x()) 
            fall_direction_ = FALL_RIGHT;
        else if(imu_data_.roll > roll_range_.y()) 
            fall_direction_ = FALL_LEFT;
        else 
            fall_direction_ = FALL_NONE;
        
        if(WM->no_power_)
        {
            init_dir_ = imu_data_.yaw;
            WM->no_power_ = false;
        }
        
        imu_data_.timestamp = CLOCK->get_timestamp();
        imu_data_.yaw = normalize_deg(imu_data_.yaw - init_dir_);

        notify(SENSOR_IMU);

        usleep(1000); // 1ms sleep
    }
}

bool Imu::start()
{
    if (!this->open())
    {
        return false;
    }

    is_open_ = true;
    is_alive_ = true;
    td_ = std::move(std::thread(&Imu::run, this));
    return true;
}

void Imu::stop()
{
    is_alive_ = false;
    is_open_ = false;
    if(fd_ >= 0)
    {
        read_end(fd_);
        fd_ = -1;
    }
}

Imu::~Imu()
{
    if (td_.joinable())
    {
        td_.join();
    }
}
