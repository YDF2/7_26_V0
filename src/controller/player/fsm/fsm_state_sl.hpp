#ifndef __FSM_STATE_SL_HPP
#define __FSM_STATE_SL_HPP

#include "fsm.hpp"
#include "vision/vision.hpp"
#include "core/worldmodel.hpp"

class FSMStateSL: public FSMState
{
public:
    FSMStateSL(FSM_Ptr fsm): FSMState(fsm)
    {
        
    }
    
    task_list OnStateEnter()
    {
        LOG(LOG_INFO)<<"Enter SL"<<endll;
        task_list tasks;
        motion::SE->search_post_end_ = false;
        VISION->localization_ = true;
        WM->localization_time_ = false;
        return tasks;
    }

    task_list OnStateExit()
    {
        LOG(LOG_INFO)<<"Exit SL"<<endll;
        task_list tasks;
        VISION->localization_ = false;
        return tasks;
    }

    task_list OnStateTick();
};

#endif
