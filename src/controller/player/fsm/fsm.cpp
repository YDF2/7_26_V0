#include "fsm_state_ready.hpp"
#include "fsm_state_getup.hpp"
#include "fsm_state_search_ball.hpp"
#include "fsm_state_goto_ball.hpp"
#include "fsm_state_kick_ball.hpp"
#include "fsm_state_dribble.hpp"
#include "fsm_state_sl.hpp"
#include "core/worldmodel.hpp"
#include "core/clock.hpp"
#include "robot/robot.hpp"
#include "skill/skill.hpp"

using namespace std;
using namespace Eigen;
using namespace motion;
using namespace robot_math;
using namespace robot;
// bool kicked = false;

task_list FSMStateSearchBall::OnStateTick()
{
    task_list tasks;
    if (WM->fall_data() != FALL_NONE)        // 若已摔倒
        return fsm_->Trans(FSM_STATE_GETUP); //状态转移
    if (WM->localization_time_)
        return fsm_->Trans(FSM_STATE_SL); //定位时间到，进入寻找球门定位状态
    // LOG(LOG_HIGH) << "----------kicked?-------\t" << kicked << endll;
    ball_block ball = WM->ball();
    if (ball.can_see)
    {
        float dir = azimuth_deg(ball.self);
        // LOG(LOG_INFO) << "ball dir:" << dir << endll;
        if (fabs(dir) < can_goto_dir_)
            return fsm_->Trans(FSM_STATE_GOTO_BALL);
        else
        {
            tasks.push_back(make_shared<WalkTask>(0.0, 0.0, dir, true));
            return tasks;
        }
    }

    if (first_in_) //首次进入状态，执行站立找球动作
    {
        first_in_ = false;
        tasks.push_back(std::make_shared<LookTask>(motion::HEAD_STATE_SEARCH_BALL));
        tasks.push_back(std::make_shared<WalkTask>(0.0, 0.0, 0.0, false));
    }
    else
    {
        // if (!ball.can_see && kicked) //转圈找球
        // {
        //     // LOG(LOG_INFO) << "向前找球！" << endll;

        //     tasks.push_back(std::make_shared<LookTask>(motion::HEAD_STATE_SEARCH_BALL));
        //     tasks.push_back(std::make_shared<WalkTask>(0.035, 0.0, 0.0, true));
        //     // if (motion::SE->search_ball_end_)
        //     //     kicked = false;
        // }
        // else if (motion::SE->search_ball_end_ && !kicked)
        // {
        //     // static double first_dir = WM->self().dir;
        //     // double rot_dir = WM->self().dir;
        //     // double bias_dir =
        //     tasks.push_back(std::make_shared<LookTask>(motion::HEAD_STATE_SEARCH_BALL));
        //     tasks.push_back(std::make_shared<WalkTask>(0.0, 0.0, 10.0, true));
        // }
        if (motion::SE->search_ball_end_) //转圈找球
        {
            tasks.push_back(std::make_shared<LookTask>(motion::HEAD_STATE_SEARCH_BALL));
           
            tasks.push_back(std::make_shared<WalkTask>(0.0, 0.0, 8.0, true));
        }
    }
    return tasks;
}

task_list FSMStateGotoBall::OnStateTick()
{
    task_list tasks;
    if (WM->fall_data() != FALL_NONE)
        return fsm_->Trans(FSM_STATE_GETUP);
    ball_block ball = WM->ball();
    self_block self = WM->self();

    if (!ball.can_see)
        return fsm_->Trans(FSM_STATE_SEARCH_BALL);
    if (WM->localization_time_)
        return fsm_->Trans(FSM_STATE_SL);

    /*******************JWJ change start ********************/
    vector<double>head_degs = ROBOT->get_head_degs();//建立vector获取头部信息
    float ball_dir = azimuth_deg(ball.self);//计算球的方位角，deg
    
    LOG(LOG_WARN)<<"ball dir ="<<azimuth_deg(ball.self)<<endl;
    
    /**************************JWJ change end*************************/
    float ball_dis = ball.self.norm();
    double self2left_dir = azimuth_deg(WM->opp_post_left - self.global); //在worldmodel.cpp修改，机器人相对左门柱的方位角
    double self2right_dir = azimuth_deg(WM->opp_post_right - self.global);//机器人相对右门柱的方位角
    // double self2mid_dir = azimuth_deg(WM->opp_post_mid - self.global);        //朝向中心方向

    // LOG(LOG_INFO) << "opp_post_left:" << WM->opp_post_left<< endl;
    if (ball_dis <= enter_kick_dis_ && self.dir >= self2right_dir && self.dir <= self2left_dir)
    {
        // LOG(LOG_INFO) << "FSMStateGotoBall: if1";
        return fsm_->Trans(FSM_STATE_KICK_BALL);
    }
    else if (ball_dis > enter_kick_dis_) //未到踢球距离时，先调整转向走向球
    {
        // LOG(LOG_INFO) << "FSMStateGotoBall: else if-2";
        // if (fabs(ball_dir) > 10.0)
        //   tasks.push_back(make_shared<WalkTask>(0.0, 0.0, ball_dir, true));                               //转向修改
        if (ball_dir > 10)
            tasks.push_back(make_shared<WalkTask>(0.0, 0.00, 8, true));
        else if (ball_dir < -10)
            tasks.push_back(make_shared<WalkTask>(0.0, 0.00, -8, true));
        else
            tasks.push_back(make_shared<WalkTask>(0.04, 0.0, 0.0, true));

    }
    else if (self.dir > self2left_dir) //调整自身方向使其面向球门
    {
        //LOG(LOG_INFO) << "FSMStateGotoBall:Turn to Goal@right_turn"<<endl;
        tasks.push_back(make_shared<WalkTask>(0.00, 0.01, -8.0, true)); //-0.01, 0.01, -8.0  before
    }
    else if (self.dir < self2right_dir)
    {
       // LOG(LOG_INFO) << "FSMStateGotoBall:Turn to Goal@left_turn"<<endl;
        tasks.push_back(make_shared<WalkTask>(0.00, -0.01, 8.0, true)); //-0.01, -0.01, 8.0 before
    }

    if (ball.self.x() < retreat_x_dis_ && fabs(ball.self.y()) > retreat_y_dis_) //离球太近后退
    {
         //LOG(LOG_INFO) << "FSMStateGotoBall:Too near,back!"<<endl;
        tasks.clear();
        tasks.push_back(make_shared<WalkTask>(-0.02, 0.0, 0.0, true));
    }
    return tasks;
}

task_list FSMStateKickBall::OnStateTick()
{
    task_list tasks;
    if (WM->fall_data() != FALL_NONE)
        return fsm_->Trans(FSM_STATE_GETUP);
    ball_block ball = WM->ball();
    self_block self = WM->self();
    if (!ball.can_see) // 若看不到球则找球
        return fsm_->Trans(FSM_STATE_SEARCH_BALL);

    float ball_dir = azimuth_deg(ball.self);
    float ball_dis = ball.self.norm();
    // LOG(LOG_INFO) << "alpha:" << ball.alpha << "\tbeta" << ball.beta << endll;
    double self2left_dir = azimuth_deg(WM->opp_post_left - self.global) ; // new
    double self2right_dir = azimuth_deg(WM->opp_post_right - self.global) ;
    double self2mid_dir = azimuth_deg(WM->opp_post_mid); //朝向中心
    if (ball_dis > exit_kick_dis_)
    { // 若离球太远则向球走
        return fsm_->Trans(FSM_STATE_GOTO_BALL);
       //  LOG(LOG_INFO) << "FSMStateKickBall: 离球太远";
    }

    if (ball.beta > retreat_beta_ && fabs(ball.alpha) > retreat_alpha_)
    { // 若离球太近
        tasks.push_back(make_shared<WalkTask>(-0.015, 0.0, 0.0, true));
        LOG(LOG_INFO) << "FSMStateKickBall: 离球太近" << endl;
        // LOG(LOG_INFO) << "FSMStateKickBall: 离球太近--if";
    }
    else
    {
        //调整方向
        if (self.dir > self2left_dir)
        {
            tasks.push_back(make_shared<WalkTask>(-0.005, 0.00, -5.0, true));
          //  LOG(LOG_INFO)<<"self2left_dir"<<self2left_dir<<endl;
          //  LOG(LOG_INFO) << "FSMStateKickBall: right转向，调整方向" << endl;
        }
        else if (self.dir < self2right_dir)
        {
           // LOG(LOG_INFO)<<"self2right_dir"<<self2right_dir<<endl;
            tasks.push_back(make_shared<WalkTask>(-0.005, 0.00, 5.0, true));
           // LOG(LOG_INFO) << "FSMStateKickBall: left转向，调整方向" << endl;
        }
        //调整左右
        else if (ball.alpha > fine_tune_alpha_max_ && self.dir < self2left_dir && self.dir > self2right_dir)
        {
           /* if(ball.beta>0.50)
            {
                tasks.push_back(std::make_shared<WalkTask>(-0.01, 0.0, 0.0, true));
                 LOG(LOG_INFO) << "FSMStateKickBall: 后退" << endl;
            }*/
            tasks.push_back(std::make_shared<WalkTask>(0.0, -0.01, 0, true));
           // LOG(LOG_INFO) << "FSMStateKickBall: right横移，调整左右" << endl;
        }
        else if (ball.alpha < fine_tune_alpha_min_ && self.dir < self2left_dir && self.dir > self2right_dir)
        {
           /* if(ball.beta>0.50)
            {
                tasks.push_back(std::make_shared<WalkTask>(-0.01, 0.0, 0.0, true));
                 LOG(LOG_INFO) << "FSMStateKickBall: 后退" << endl;
            }*/
            tasks.push_back(std::make_shared<WalkTask>(0.0, 0.01, 0, true));
          //  LOG(LOG_INFO) << "FSMStateKickBall: left横移，调整左右" << endl;
        }
        else
        {
            //调整前后
            if (ball.beta < fine_tune_beta_min_) // ball.beta < 0.39 BEFORE
            {
                tasks.push_back(std::make_shared<WalkTask>(0.015, 0.0, 0, true));
            //     LOG(LOG_INFO) << "FSMStateKickBall: 前进" << endl;
            }
            else if (ball.beta > fine_tune_beta_max_) // 45before
            {
                tasks.push_back(std::make_shared<WalkTask>(-0.01, 0.0, 0.0, true));
              //  LOG(LOG_INFO) << "FSMStateKickBall: 后退" << endl;
            }
            else
            {
                // kicked = true;
                tasks.push_back(std::make_shared<ActionTask>("left_little_kick"));
                LOG(LOG_INFO) << "FSMStateKickBall: 踢球" << endl;
                //return fsm_->Trans(FSM_STATE_DRIBBLE);
            }
            LOG(LOG_WARN)<<"ball_alpha="<<ball.alpha<<endl;
            LOG(LOG_WARN)<<"ball_beta="<<ball.beta<<endl;
        }
    }
    return tasks;
}

task_list FSMStateDribble::OnStateTick()
{
    task_list tasks;
    if (WM->fall_data() != FALL_NONE)
        return fsm_->Trans(FSM_STATE_GETUP);
    ball_block ball = WM->ball();
    if (!ball.can_see)
        return fsm_->Trans(FSM_STATE_SEARCH_BALL);
    if (WM->localization_time_)
        return fsm_->Trans(FSM_STATE_SL);

    float ball_dir = azimuth_deg(ball.self);
    float ball_dis = ball.self.norm();
    if (ball_dis > exit_kick_dis_)
        return fsm_->Trans(FSM_STATE_GOTO_BALL);

    if (ball.beta > retreat_beta_ && fabs(ball.alpha) > retreat_alpha_)
        tasks.push_back(make_shared<WalkTask>(-0.02, 0.0, 0.0, true));
    else
    {
        if (ball.alpha > 0.1)
            tasks.push_back(std::make_shared<WalkTask>(0.0, -0.01, 0.0, true));
        else if (ball.alpha < -0.1)
            tasks.push_back(std::make_shared<WalkTask>(0.0, 0.01, 0.0, true));
        else
        {
            float target_dir = azimuth_deg(target_pos_ - WM->self().global);
            float deg = normalize_deg(WM->self().dir - target_dir);
            if (fabs(deg) > 15.0)
                tasks.push_back(std::make_shared<WalkTask>(0.0, 0.0, -deg, true));
            else
                tasks.push_back(std::make_shared<WalkTask>(0.035, 0.0, 0.0, true));
        }
    }
    return tasks;
}

task_list FSMStateSL::OnStateTick() //寻找球门定位，定位时间在player.cpp
{
    task_list tasks;
    if (WM->fall_data() != FALL_NONE)
        return fsm_->Trans(FSM_STATE_GETUP);
    if (motion::SE->search_post_end_)
        return fsm_->Trans(FSM_STATE_SEARCH_BALL);

    tasks.push_back(std::make_shared<WalkTask>(0.0, 0.0, 0.0, false));
    tasks.push_back(std::make_shared<LookTask>(motion::HEAD_STATE_SEARCH_POST)); //执行寻找球门动作，该动作在scan_engine.cpp

    return tasks;
}

task_list FSMStateGetup::OnStateTick()
{
    task_list tasks;
    if (WM->fall_data() != FALL_NONE)
    {
        // kicked = false;
        if (WM->fall_data() == FALL_FORWARD)
            tasks.push_back(std::make_shared<ActionTask>("front_getup"));
        else if (WM->fall_data() == FALL_BACKWARD)
            tasks.push_back(std::make_shared<ActionTask>("back_getup"));
        else if (WM->fall_data() == FALL_LEFT)
            tasks.push_back(std::make_shared<ActionTask>("right_arm"));
        else
            tasks.push_back(std::make_shared<ActionTask>("left_arm"));
    }
    else
    {
        tasks = fsm_->Trans(FSM_STATE_READY);
    }
    return tasks;
}

task_list FSMStateReady::OnStateTick()
{
    task_list tasks, tlist;
    if (WM->fall_data() != FALL_NONE)
        return fsm_->Trans(FSM_STATE_GETUP);
    tasks.push_back(std::make_shared<WalkTask>(0.0, 0.0, 0.0, false));
    tasks.push_back(std::make_shared<LookTask>(0.0, 40.0));
    tlist = fsm_->Trans(FSM_STATE_SEARCH_BALL);
    tasks.insert(tasks.end(), tlist.begin(), tlist.end());
    return tasks;
}
