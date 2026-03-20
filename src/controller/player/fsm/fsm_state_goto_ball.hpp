#ifndef __FSM_STATE_GOTO_BALL_HPP
#define __FSM_STATE_GOTO_BALL_HPP

#include "fsm.hpp"


class FSMStateGotoBall: public FSMState
{
public:
    FSMStateGotoBall(FSM_Ptr fsm): FSMState(fsm)
    {
        enter_kick_dis_ = 0.3;
        retreat_x_dis_ = 0.2;
        retreat_y_dis_ = 0.1;
    }
    
    task_list OnStateEnter()
    {
        LOG(LOG_INFO)<<"Enter goto ball"<<endll;
        task_list tasks;
        return tasks;
    }

    task_list OnStateExit()
    {
        LOG(LOG_INFO)<<"Exit goto ball"<<endll;
        task_list tasks;
        return tasks;
    }

    task_list OnStateTick();

private:
    float enter_kick_dis_;
    float retreat_x_dis_;
    float retreat_y_dis_;
};

#endif
