#ifndef __FSM_STATE_READY_HPP
#define __FSM_STATE_READY_HPP

#include "fsm.hpp"

class FSMStateReady : public FSMState
{
public:
    FSMStateReady(FSM_Ptr fsm) : FSMState(fsm) {}

    task_list OnStateEnter()
    {
        LOG(LOG_INFO) << "Enter ready" << endll;
        task_list tasks;
        return tasks;
    }

    task_list OnStateExit()
    {
        LOG(LOG_INFO) << "Exit ready" << endll;
        task_list tasks;
        return tasks;
    }

    task_list OnStateTick();

private:
    bool dribble_fin = false;
    float exit_kick_dis_ = 0.45;
    float retreat_alpha_ = 0.30;
    float retreat_beta_ = 0.45;

    Eigen::Vector2d target_pos_;
};

#endif
