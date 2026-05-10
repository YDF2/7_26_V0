#ifndef __BUTTON_HPP
#define __BUTTON_HPP

#include <thread>
#include <memory>
#include <atomic>
#include "sensor.hpp"
#include "drivers/gpio/gpio.hpp"

class Button: public Sensor
{
public:
    Button();
    ~Button();
    bool start();
    void stop();

    inline bool button_1()
    {
        return bt1_status_;
    }

    inline bool button_2()
    {
        return bt2_status_;
    }

private:
    void run();
    std::thread td_;
    bool can_use_;
    std::atomic_bool bt1_status_;
    std::atomic_bool bt2_status_;
    std::shared_ptr<Gpio> button1_;
    std::shared_ptr<Gpio> button2_;
};

#endif