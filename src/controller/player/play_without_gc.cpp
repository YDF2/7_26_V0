#include <list>
#include "player.hpp"
#include "task/look_task.hpp"
#include "task/walk_task.hpp"
#include "task/action_task.hpp"
#include "core/worldmodel.hpp"
#include "configuration.hpp"
#include "math/math.hpp"
#include "vision/vision.hpp"

using namespace std;
using namespace robot;
using namespace Eigen;
using namespace robot_math;
using namespace motion;


std::list<task_ptr> Player::play_without_gc()
{
    list<task_ptr> tasks;
    if(!played_)
        played_ = true;
    tasks = fsm_->Tick();
    return tasks;
}