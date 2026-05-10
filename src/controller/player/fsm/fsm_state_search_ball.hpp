#ifndef __FSM_STATE_SEARCH_BALL_HPP
#define __FSM_STATE_SEARCH_BALL_HPP

#include "fsm.hpp"
#include <vector>
#include <eigen3/Eigen/Dense>
#include "math/math.hpp"

class FSMStateSearchBall: public FSMState
{
public:
    FSMStateSearchBall(FSM_Ptr fsm): FSMState(fsm)
    {
        can_goto_dir_ = 70.0;
    }
    
    task_list OnStateEnter()
    {
        LOG(LOG_INFO)<< "Enter serach ball"<<endll;
        task_list tasks;
        init_search_dir_ = WM->self().dir;
        last_search_dir_ = init_search_dir_;
        first_in_ = true;
        motion::SE->search_ball_end_ = false;
        
        return tasks;
    }

    task_list OnStateExit()
    {
        LOG(LOG_INFO)<< "Exit serach ball"<<endll;
        task_list tasks;
        return tasks;
    }

    task_list OnStateTick();

private:
    float last_search_dir_;
    float init_search_dir_;
    float can_goto_dir_;
    bool first_in_;
};

#endif
