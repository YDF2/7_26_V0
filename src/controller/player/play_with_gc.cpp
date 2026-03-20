#include <list>
#include "player.hpp"
#include "task/look_task.hpp"
#include "task/walk_task.hpp"
#include "task/action_task.hpp"
#include "core/worldmodel.hpp"
#include "configuration.hpp"
#include "math/math.hpp"
#include "vision/vision.hpp"
#include "skill/skill.hpp"

using namespace std;
using namespace robot;
using namespace Eigen;
using namespace robot_math;
using namespace motion;

std::list<task_ptr> Player::play_with_gc()
{
    list<task_ptr> tasks;
    RoboCupGameControlData gc_data = WM->gc_data();
    Vector2f head_init = SE->head_init_deg_;
    int secondT = (int)gc_data.secondaryTime;
    int kickoff = (int)gc_data.kickOffTeam;
    int team_index = gc_data.teams[0].teamNumber == CONF->team_number() ? 0 : 1;

    switch (gc_data.state) //位置信息在config文件中修改
    {
    case STATE_INITIAL:
       LOG(LOG_WARN) <<"SETTING INIT POS"<<endl;
       LOG(LOG_HIGH)<<"My_Team_Number="<<kickoff<<endl;
        if (kickoff == CONF->team_number() || kickoff == DROPBALL) //作为进攻方设置入场位置，teamnumber在config文件中修改
        {
            if (CONF->get_my_role() == "front")
            {
                /*if(kickoff == CONF->team_number())
                    WM->set_my_pos(kickoff_pos_);
                else
                    WM->set_my_pos(init_pos_);*/
                LOG(LOG_DEBUG) << "DIR:" << WM->self().dir << endll; //新加
                if (WM->self().dir > 45.0)
                {
                    WM->set_my_pos(Vector2d(start_pos_.x(), -start_pos_.y()));
                }
                else
                {
                    WM->set_my_pos(start_pos_);
                }
            }
            else if (CONF->get_my_role() == "guard")
            {
                if (WM->self().dir > 45.0)
                    WM->set_my_pos(Vector2d(start_pos_.x(), -start_pos_.y()));
                else
                    WM->set_my_pos(start_pos_);
            }
            else if (CONF->get_my_role() == "keeper")
            {
                WM->set_my_pos(init_pos_);
            }
        }
        else //作为防守方设置入场位置
        {
            if (WM->self().dir > 45.0)
                WM->set_my_pos(Vector2d(start_pos_.x(), -start_pos_.y()));
            else
                WM->set_my_pos(start_pos_);
        }
        tasks.push_back(make_shared<WalkTask>(0.0, 0.0, 0.0, false));
        tasks.push_back(make_shared<LookTask>(head_init[0], head_init[1]));
        break;
    case STATE_READY:
        if (kickoff == CONF->team_number() || kickoff == DROPBALL) //作为进攻方执行入场动作，或进球后回到初始位置
        {
            LOG(LOG_WARN)<<"进攻方"<<endl;
            if (CONF->get_my_role() == "front")
            {
                // LOG(LOG_INFO) << "played:" << played_ << endll;
                // if (!played_)
                // {
                //     if (kickoff == CONF->team_number())
                //         WM->set_my_pos(kickoff_pos_);
                //     else
                //         WM->set_my_pos(init_pos_);
                // }
                // else
                // LOG(LOG_INFO) << "..........ready_attact_front:..........." << endll;
                LOG(LOG_WARN)<<"GO TO READYPOS_FRONT"<<endl; 
                tasks.push_back(skill_goto(WM->self(), kickoff_pos_, 0.0));
            }
            else if (CONF->get_my_role() == "guard")
            {
                tasks.push_back(skill_goto(WM->self(), init_pos_, 0.0));
                LOG(LOG_WARN)<<"GO TO READYPOS_GUARD"<<endl;
            }
            else if (CONF->get_my_role() == "keeper")
            {
                WM->set_my_pos(init_pos_);
            }
        }
        else //作为防守方执行入场动作，或进球后回到初始位置
        {
            LOG(LOG_WARN)<<"防守方"<<endl;
            if (CONF->get_my_role() == "front")
            {
                // LOG(LOG_INFO) << "...........ready_defence_front............." << endll;
                tasks.push_back(skill_goto(WM->self(), init_pos_, 0.0));
            }
            else if (CONF->get_my_role() == "guard")
            {
                tasks.push_back(skill_goto(WM->self(), init_pos_, 0.0));
            }
            else if (CONF->get_my_role() == "keeper")
            {
                WM->set_my_pos(init_pos_);
            }
        }
        tasks.push_back(make_shared<LookTask>(head_init[0], head_init[1]));
        break;
    case STATE_SET:
        tasks.push_back(make_shared<WalkTask>(0.0, 0.0, 0.0, false));
        tasks.push_back(make_shared<LookTask>(head_init[0], head_init[1]));
        break;
    case STATE_PLAYING:
        played_ = true;
        // LOG(LOG_INFO) << "team_index" << team_index << "\tid" << CONF->id() << endll;
        // LOG(LOG_INFO) << "penalty_0_team_0:     " << gc_data.teams[0].players[CONF->id() - 1].penalty << endll;
        // LOG(LOG_INFO) << "penalty_0_team_1:     " << gc_data.teams[0].players[CONF->id()].penalty << endll;
        // LOG(LOG_INFO) << "penalty_1_team_0:     " << gc_data.teams[1].players[CONF->id() - 1].penalty << endll;
        // LOG(LOG_INFO) << "penalty_1_team_1:     " << gc_data.teams[1].players[CONF->id()].penalty << endll;
        if (gc_data.teams[team_index].players[CONF->id() - 1].penalty == HL_PICKUP_OR_INCAPABLE || gc_data.teams[team_index].players[CONF->id() - 1].penalty == HL_SERVICE)
        {
            LOG(LOG_INFO) << "...........play_pickup_front............." << endll;
            tasks.push_back(make_shared<LookTask>(HEAD_STATE_SEARCH_BALL));
            tasks.push_back(make_shared<WalkTask>(0.0, 0.0, 0.0, false));
            if (WM->self().dir > 45.0)
                WM->set_my_pos(Vector2d(pickup_pos_.x(), -pickup_pos_.y())); //
            else if (WM->self().dir < -45.0)
                WM->set_my_pos(pickup_pos_); //
        }
        else
        {
            if (kickoff != DROPBALL && kickoff != CONF->team_number() && secondT != 0)
            {
                LOG(LOG_INFO) << "...........play_search_front............." << endll;
                tasks.push_back(make_shared<LookTask>(HEAD_STATE_SEARCH_BALL));
            }
            else
            {
                // LOG(LOG_INFO) << "...........play_kickoff_normally............." << endll;
                task_list tlist = fsm_->Tick();
                tasks.insert(tasks.end(), tlist.begin(), tlist.end());
            }
        }
        break;
    case STATE_FINISHED:
        tasks.push_back(make_shared<WalkTask>(0.0, 0.0, 0.0, false));
        tasks.push_back(make_shared<LookTask>(head_init[0], head_init[1]));
        break;
    default:
        tasks.push_back(make_shared<WalkTask>(0.0, 0.0, 0.0, false));
        tasks.push_back(make_shared<LookTask>(head_init[0], head_init[1]));
        break;
    }
    return tasks;
}