#ifndef __LED_TASK_HPP
#define __LED_TASK_HPP

#include "task.hpp"
#include "engine/led/led_engine.hpp"

class LedTask: public Task
{
public:
    LedTask(bool led1, bool led2): led1_(led1), led2_(led2), Task("led")
    {

    }

    bool perform()
    {
        LE->set_state(LED_CUSTOM);
        LE->set_led(1, led1_);
        LE->set_led(2, led2_);
        return true;
    }

private:
    bool led1_;
    bool led2_;
};

#endif
