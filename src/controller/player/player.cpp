#include "player.hpp"
#include "configuration.hpp"
#include "task/action_task.hpp"
#include "task/look_task.hpp"
#include "task/walk_task.hpp"
#include "task/gcret_task.hpp"
#include "task/say_task.hpp"
#include "server/server.hpp"
#include "core/adapter.hpp"
#include "core/clock.hpp"
#include "skill/skill.hpp"
#include "engine/walk/walk_engine.hpp"
#include "engine/scan/scan_engine.hpp"
#include "engine/action/action_engine.hpp"
#include "engine/led/led_engine.hpp"
#include "fsm/fsm_state_getup.hpp"
#include "fsm/fsm_state_goto_ball.hpp"
#include "fsm/fsm_state_kick_ball.hpp"
#include "fsm/fsm_state_ready.hpp"
#include "fsm/fsm_state_search_ball.hpp"
#include "fsm/fsm_state_sl.hpp"
#include "fsm/fsm_state_dribble.hpp"

using namespace std;
using namespace motion;
using namespace Eigen;

Player::Player() : Timer(CONF->get_config_value<int>("think_period"))
{
    is_alive_ = false;
    period_count_ = 0;
    btn_count_ = 0;
    role_ = CONF->get_my_role();
    init_pos_ = CONF->get_config_vector<double, 2>("strategy." + role_ + ".init_pos");
    start_pos_ = CONF->get_config_vector<double, 2>("strategy." + role_ + ".start_pos");
    pickup_pos_ = CONF->get_config_vector<double, 2>("strategy." + role_ + ".pickup_pos");
    if (role_ == "front")
        kickoff_pos_ = CONF->get_config_vector<double, 2>("strategy." + role_ + ".kickoff_pos");

    self_location_count_ = 0;
    played_ = false;
    fsm_ = make_shared<FSM>();
}

void Player::run()
{
    if (is_alive_)
    {
        period_count_++;

        if (OPTS->use_robot())
        {
            if (CLOCK->get_timestamp() - WM->imu_data().timestamp > 500)
                WM->no_power_ = true;
            if (CLOCK->get_timestamp() - WM->imu_data().timestamp > 3000 && WM->imu_data().timestamp != 0)
            {
                LOG(LOG_ERROR) << "motor power has been turned off!" << endll;
                raise(SIGINT);
            }
        }
        if (OPTS->use_remote())
        {
            play_with_remote();
        }
        else
        {
            list<task_ptr> tasks, tlist;
            if ((period_count_ * period_ms_ / 100) % 5 == 0)
            {
                if (OPTS->use_gc())
                    tasks.push_back(make_shared<GcretTask>());
                if (OPTS->use_comm())
                    tasks.push_back(make_shared<SayTask>(fsm_->get_state()));
            }
            if ((period_count_ * period_ms_ / 100) % 20 == 0)
            {
                WM->reset_hear_info();
            }

            tlist = think();
            tasks.insert(tasks.end(), tlist.begin(), tlist.end());
            for (auto &tsk : tasks)
            {
                if (tsk.get())
                    tsk->perform();
            }
        }
    }
}

list<task_ptr> Player::think()
{
    list<task_ptr> tasks, tlists;
    //LOG(LOG_HIGH) << "im thinking" << endll;
    if (OPTS->image_record())
    {
        tasks.push_back(make_shared<LookTask>(HEAD_STATE_SEARCH_BALL));
        tasks.push_back(make_shared<WalkTask>(0.0, 0.0, 0.0, true));
    }
    else
    {
        if (OPTS->kick_mode() == Options::KICK_NORMAL)
        {
            if (played_)
                self_location_count_++;
            if (self_location_count_ == 600) //寻找球门时间
            {
                WM->localization_time_ = true;
                self_location_count_ = 0;
            }
            if (OPTS->use_gc())
            {
                tlists = play_with_gc();
            }
            else
            {
                tlists = play_without_gc();
            }
        }
        else if (OPTS->kick_mode() == Options::KICK_PENALTY)
        {
            LOG(LOG_HIGH) << "in penalty  " << endll; 
            static bool start_penalty = false;
            static bool left = true;
            static float init_dir = 0.0;
            tasks.push_back(make_shared<LookTask>(0.0, 60.0));



            if (CONF->id()==1)
            {
                start_penalty = true;
                LOG(LOG_HIGH) << "button11111  " << endll;
                left = true;
                init_dir = WM->self().dir;
            }
            else if (CONF->id()==2)
            {
		LOG(LOG_HIGH) << "button2222  " << endll;
                start_penalty = true;
                left = false;
                init_dir = WM->self().dir;
            }
            if (start_penalty){
		LOG(LOG_HIGH) << "success  " << endll;
                tasks.push_back(skill_penalty_kick(WM->ball()));}
        }
    }

    tasks.insert(tasks.end(), tlists.begin(), tlists.end());
    return tasks;
}

bool Player::init()
{
    fsm_->Register(FSM_STATE_READY, make_shared<FSMStateReady>(fsm_));
    fsm_->Register(FSM_STATE_GETUP, make_shared<FSMStateGetup>(fsm_));
    fsm_->Register(FSM_STATE_SEARCH_BALL, make_shared<FSMStateSearchBall>(fsm_));
    fsm_->Register(FSM_STATE_GOTO_BALL, make_shared<FSMStateGotoBall>(fsm_));
    fsm_->Register(FSM_STATE_KICK_BALL, make_shared<FSMStateKickBall>(fsm_));
    fsm_->Register(FSM_STATE_SL, make_shared<FSMStateSL>(fsm_));
    fsm_->set_state(FSM_STATE_READY);

    WM->set_my_pos(init_pos_);

    if (OPTS->use_debug())
        SERVER->start();
    CLOCK->start();

    if (!regist())
    {
        return false;
    }
    is_alive_ = true;

    if (OPTS->use_robot())
    {
        while (!dynamic_pointer_cast<Motor>(sensors_["motor"])->is_connected() && is_alive_)
        {
            LOG(LOG_WARN) << "waiting for motor connection, please turn on the power." << endll;
            sleep(1);
        }

        if (!is_alive_)
        {
            return true;
        }
    }

    MADT->start();
    WE->start();
    SE->start();
    AE->start();
    if (OPTS->use_robot())
    {
        LE->start();
    }
    ActionTask p("ready");
    p.perform();
    start_timer();
    return true;
}

void Player::stop()
{
    is_alive_ = false;
    WE->stop();
    SE->stop();
    AE->stop();
    if (OPTS->use_robot())
    {
        LE->stop();
    }
    MADT->stop();

    if (is_alive_)
    {
        delete_timer();
    }

    sleep(1);
    unregist();
    sleep(1);
    if (OPTS->use_debug())
        SERVER->stop();
    CLOCK->stop();
}

bool Player::regist()
{
    sensors_.clear();

    if (OPTS->use_camera())
    {
        sensors_["camera"] = std::make_shared<Camera>();
        sensors_["camera"]->attach(VISION);
        sensors_["camera"]->start();
        if (!VISION->start())
        {
            return false;
        }
    }

    sensors_["motor"] = std::make_shared<Motor>();
    sensors_["motor"]->attach(WE);

    if (!sensors_["motor"]->start())
    {
        return false;
    }

    if (OPTS->use_robot())
    {
        sensors_["imu"] = std::make_shared<Imu>();
        sensors_["imu"]->attach(WM);
        sensors_["imu"]->attach(VISION);
        sensors_["imu"]->attach(WE);
        sensors_["imu"]->start();

        sensors_["button"] = std::make_shared<Button>();
        sensors_["button"]->attach(WM);
        sensors_["button"]->start();
    }

    if (OPTS->use_gc())
    {
        sensors_["gamectrl"] = std::make_shared<GameCtrl>();
        sensors_["gamectrl"]->attach(WM);
        sensors_["gamectrl"]->start();
    }

    if (OPTS->use_comm())
    {
        sensors_["hear"] = std::make_shared<Hear>();
        sensors_["hear"]->attach(WM);
        sensors_["hear"]->start();
    }

    return true;
}

void Player::unregist()
{
    if (sensors_.find("button") != sensors_.end())
    {
        sensors_["button"]->detach(WM);
        sensors_["button"]->stop();
    }
    if (sensors_.find("imu") != sensors_.end())
    {
        sensors_["imu"]->detach(WM);
        sensors_["imu"]->detach(VISION);
        sensors_["imu"]->detach(WE);
        sensors_["imu"]->stop();
    }
    if (sensors_.find("motor") != sensors_.end())
    {
        sensors_["motor"]->detach(WE);
        sensors_["motor"]->stop();
    }
    if (sensors_.find("camera") != sensors_.end())
    {
        sensors_["camera"]->detach(VISION);
        sensors_["camera"]->stop();
        VISION->stop();
    }
    if (sensors_.find("gamectrl") != sensors_.end())
    {
        sensors_["gamectrl"]->detach(WM);
        sensors_["gamectrl"]->stop();
    }
    if (sensors_.find("hear") != sensors_.end())
    {
        sensors_["hear"]->detach(WM);
        sensors_["hear"]->stop();
    }
}

sensor_ptr Player::get_sensor(const std::string &name)
{
    auto iter = sensors_.find(name);

    if (iter != sensors_.end())
    {
        return iter->second;
    }

    return nullptr;
}
