#include "worldmodel.hpp"
#include "localization/soccermap.h"

using namespace Eigen;
using namespace robot_math;
using namespace robot;
using namespace std;

WorldModel::WorldModel()
{
    fall_direction_ = FALL_NONE;
    support_foot_ = robot::DOUBLE_SUPPORT;
    player_infos_[CONF->id()].id = CONF->id();
    opp_post_left = Vector2d(SOCCERMAP->width()/200.0, SOCCERMAP->goalWidth()/200.0 - 0.3);         //
    opp_post_right = Vector2d(SOCCERMAP->width()/200.0, -SOCCERMAP->goalWidth()/200.0 - 0.1);        // 
    opp_post_mid = Vector2d(4.5,0);
    localization_time_ = false;
    no_power_ = true;
}

void WorldModel::updata(const pub_ptr &pub, const int &type)
{
    if (type == Sensor::SENSOR_IMU)
    {
        std::shared_ptr<Imu> sptr = std::dynamic_pointer_cast<Imu>(pub);
        imu_mtx_.lock();
        imu_data_ = sptr->data();
        fall_direction_ = sptr->fall_direction();
        imu_mtx_.unlock();
        self_mtx_.lock();
        self_block_.dir = sptr->data().yaw;
        self_mtx_.unlock();  
        info_mtx_.lock();
        player_infos_[CONF->id()].dir = sptr->data().yaw;
        info_mtx_.unlock();        
        return;
    }

    if(type == Sensor::SENSOR_BUTTON)
    {
        std::shared_ptr<Button> sptr = std::dynamic_pointer_cast<Button>(pub);
        bt1_status_ = sptr->button_1();
        bt2_status_ = sptr->button_2();
        return;
    }

    if(type == Sensor::SENSOR_GC)
    {
        std::shared_ptr<GameCtrl> sptr = std::dynamic_pointer_cast<GameCtrl>(pub);
        gc_mtx_.lock();
        gc_data_ = sptr->data();
        gc_mtx_.unlock();
        return;
    }

    if(type == Sensor::SENSOR_HEAR)
    {
        std::shared_ptr<Hear> sptr = std::dynamic_pointer_cast<Hear>(pub);
        info_mtx_.lock();
        player_info info = sptr->info();
        player_infos_[info.id] = info;
        info_mtx_.unlock();
        return;
    }
}

void WorldModel::set_my_pos(const Eigen::Vector2d &my)
{
    info_mtx_.lock();
    player_infos_[CONF->id()].x = my.x();
    player_infos_[CONF->id()].y = my.y();
    info_mtx_.unlock();
    self_mtx_.lock();
    self_block_.global = my;
    self_mtx_.unlock();
}

void WorldModel::set_ball_pos(const Eigen::Vector2d &global, const Eigen::Vector2d &my, const Eigen::Vector2i &pix,
    float alpha, float beta,  bool can)
{
    info_mtx_.lock();
    player_infos_[CONF->id()].ball_x = global.x();
    player_infos_[CONF->id()].ball_y = global.y();
    player_infos_[CONF->id()].can_see = can;
    info_mtx_.unlock();
    
    ball_mtx_.lock();
    ball_block_.global = global;
    ball_block_.self = my;
    ball_block_.pixel = pix;
    ball_block_.alpha = alpha;
    ball_block_.beta = beta;
    ball_block_.can_see = can;
    ball_mtx_.unlock();
}

void WorldModel::reset_hear_info()
{
    info_mtx_.lock();
    for(auto &item:player_infos_)
    {
        if(item.first != CONF->id())
            item.second.can_see = false;
    }
    info_mtx_.unlock();
}
