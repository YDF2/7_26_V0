#ifndef __FSM_STATE_KICK_BALL_HPP
#define __FSM_STATE_KICK_BALL_HPP

#include "fsm.hpp"
#include "engine/scan/scan_engine.hpp"
#include "configuration.hpp"

using namespace Eigen; //新加
class FSMStateKickBall: public FSMState
{
public:
    FSMStateKickBall(FSM_Ptr fsm): FSMState(fsm)
    {
        exit_kick_dis_ = CONF->get_config_value<float>("kick.exit_kick_dis_");
        retreat_alpha_ = CONF->get_config_value<float>("kick.retreat_alpha_");
        retreat_beta_ = CONF->get_config_value<float>("kick.retreat_beta_");//0.45before
        fine_tune_alpha_max_ = CONF->get_config_value<float>("kick.fine_tune_alpha_max");
        fine_tune_alpha_min_ = CONF->get_config_value<float>("kick.fine_tune_alpha_min");
        fine_tune_beta_max_ = CONF->get_config_value<float>("kick.fine_tune_beta_max");
        fine_tune_beta_min_ = CONF->get_config_value<float>("kick.fine_tune_beta_min");
    }
    
    task_list OnStateEnter()
    {
        LOG(LOG_INFO)<<"Enter kick ball"<<endll;
        task_list tasks;
        tasks.push_back(std::make_shared<LookTask>(0.0, motion::SE->pitch_range_[1]));
        //Vector2d X = WM->ball().global; //新加
        //Vector2d Y = WM->self().global;                //新加
        //LOG(LOG_DEBUG) <<"ball" << X(0) <<"  " << X(1) <<"self" << Y(0) <<"  " << Y(1)<<endll; //新加        
        //LOG(LOG_DEBUG) <<"self" << Y(0) <<"  " << Y(1) <<endll; //新加
        return tasks;
    }

    task_list OnStateExit()
    {
        LOG(LOG_INFO)<<"Exit kick ball"<<endll;
        task_list tasks;
        return tasks;
    }

    task_list OnStateTick();

private:
    float exit_kick_dis_;   // 踢球容许的最大离球距离，大于该值不应踢球
    float retreat_alpha_;   // 踢球容许的最小离球横向距离（绝对值），小于该值应调整
    float retreat_beta_;    // 踢球容许的最小离球纵向距离（绝对值），小于该值应调整
    float fine_tune_alpha_max_; // 微调时离球纵向距离的最大值，大于该值应微调
    float fine_tune_alpha_min_; // 微调时离球纵向距离的最小值，小于该值应微调
    float fine_tune_beta_max_;  // 微调时离球横向距离的最大值，大于该值应微调
    float fine_tune_beta_min_;  // 微调时离球横向距离的最小值，小于该值应微调
};

#endif
