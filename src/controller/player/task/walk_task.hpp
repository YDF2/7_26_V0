#ifndef __WALK_TASK_HPP
#define __WALK_TASK_HPP

#include <eigen3/Eigen/Dense>
#include "task.hpp"
#include "engine/walk/walk_engine.hpp"
#include "core/adapter.hpp"

// x：纵向步长，y：横向步长，dir：脚的转角，enable：是否执行WE
class WalkTask: public Task
{
public:
    WalkTask(float x, float y, float dir, bool enable)
        : x_(x), y_(y), d_(dir), e_(enable), Task("walk")
    {

    }

    bool perform()
    {
        //MADT->mode() = adapter::MODE_WALK;
        motion::WE->set_params(x_, y_, d_, e_);
        return true;
    }
private:
    float x_, y_, d_;
    bool e_;
};

#endif
