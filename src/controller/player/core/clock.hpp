#ifndef __CLOCK_HPP
#define __CLOCK_HPP

#include <atomic>
#include "timer.hpp"
#include "singleton.hpp"
#include "configuration.hpp"

class Clock: public Timer, public Singleton<Clock>
{
public:
    Clock(): Timer(CONF->get_config_value<int>("clock_period"))
    {
        timestamp_ = 0;
    }

    // 初始化计时器
    void start()
    {
        start_timer();
    }

    void stop()
    {
        delete_timer();
    }

    // 开始计时
    void run()
    {
        timestamp_ += period_ms_;
    }

    unsigned int get_timestamp()
    {
        return timestamp_;
    }

private:
    std::atomic_int timestamp_;
};

#define CLOCK Clock::instance()

#endif
