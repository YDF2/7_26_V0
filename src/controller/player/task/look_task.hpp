#ifndef __LOOK_TASK_HPP
#define __LOOK_TASK_HPP

#include "task.hpp"
#include "engine/scan/scan_engine.hpp"
#include <vector>
#include <eigen3/Eigen/Dense>

// yaw：偏航角；pitch：俯仰角（参见欧拉角）
class LookTask: public Task
{
public:
    LookTask(float yaw, float pitch)
        :s_(motion::HEAD_STATE_LOOKAT), Task("look")
    {
        pitch_ = pitch;
        yaw_ = yaw;
    }

    LookTask(motion::Head_State state): s_(state), Task("look")
    {
        pitch_ = 0.0;
        yaw_ = 0.0;
    }

    bool perform()
    {
        motion::SE->set_params(yaw_, pitch_, s_);
        return true;
    }
private:
    float pitch_, yaw_;
    motion::Head_State s_;
};

#endif
