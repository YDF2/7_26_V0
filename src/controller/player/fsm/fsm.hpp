#ifndef __FSM_HPP
#define __FSM_HPP

#include <memory>
#include <map>
#include "model.hpp"
#include "task/task.hpp"
#include "task/walk_task.hpp"
#include "task/action_task.hpp"
#include "task/look_task.hpp"
#include "core/worldmodel.hpp"
#include "engine/scan/scan_engine.hpp"

class FSM;

typedef std::shared_ptr<FSM> FSM_Ptr;

class FSMState
{
public:
    FSMState(FSM_Ptr fsm) : fsm_(fsm) {}
    virtual task_list OnStateEnter() = 0;
    virtual task_list OnStateExit() = 0;
    virtual task_list OnStateTick() = 0;

protected:
    FSM_Ptr fsm_;
};

typedef std::shared_ptr<FSMState> FSMState_Ptr;

class FSM //状态机，统管所有状态
{
public:
    FSM()
    {
    }

    task_list Trans(fsm_state state) //状态转移
    {
        task_list tasks, tlist;
        tasks = states_[current_state_]->OnStateExit();
        tlist = states_[state]->OnStateEnter();
        current_state_ = state;
        tasks.insert(tasks.end(), tlist.begin(), tlist.end());
        return tasks;
    }

    task_list Tick() //每帧调用
    {
        return states_[current_state_]->OnStateTick();
    }

    bool Register(fsm_state s, FSMState_Ptr state) //注册新的状态
    {
        states_[s] = state;
        return true;
    }

    fsm_state get_state()
    {
        return current_state_;
    }

    void set_state(fsm_state s) //设置初始状态
    {
        current_state_ = s;
    }

private:
    std::map<int, FSMState_Ptr> states_;
    fsm_state current_state_;
};

#endif
