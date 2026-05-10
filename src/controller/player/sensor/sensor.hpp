#ifndef __SENSOR_HPP
#define __SENSOR_HPP

#include <iomanip>
#include <memory>
#include "observer.hpp"
#include <iostream>
#include "logger.hpp"

class Sensor: public Publisher
{
public:
    enum sensor_type
    {
        SENSOR_NONE = 0,
        SENSOR_BUTTON = 1,
        SENSOR_GC = 2,
        SENSOR_HEAR = 3,
        SENSOR_IMU = 4,
        SENSOR_MOTOR = 5,
        SENSOR_CAMERA = 6,
        
        SENSOR_END = 10
    };
    Sensor(const std::string &name): name_(name)
    {
        LOG(LOG_INFO) << std::setw(12) << "sensor:" << std::setw(18) << "[" + name_ + "]" << " started!" << endll;
    }

    ~Sensor()
    {
        LOG(LOG_INFO) << std::setw(12) << "sensor:" << std::setw(18) << "[" + name_ + "]" << " ended!" << endll;
    }

    virtual bool start() = 0;
    virtual void stop() = 0;

    inline bool is_open() const
    {
        return is_open_;
    }

protected:
    bool is_open_;
    bool is_alive_;
    std::string name_;
};

typedef std::shared_ptr<Sensor> sensor_ptr;

#endif
