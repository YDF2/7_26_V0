#include "action_engine.hpp"
#include "configuration.hpp"
#include "logger.hpp"
#include "robot/robot.hpp"
#include "core/adapter.hpp"
#include "engine/walk/walk_engine.hpp"
#include <unistd.h>

namespace motion
{

using namespace robot;

ActionEngine::ActionEngine()
{
    step_ = CONF->get_config_value<float>("hardware.motor.period");
    poses_.clear();
    pos_times_.clear();
    in_action_ = false;
}

ActionEngine::~ActionEngine()
{
    if (td_.joinable())
    {
        td_.join();
    }
    LOG(LOG_INFO) << std::setw(12) << "engine:" << std::setw(18) << "[ActionEngine]" << " ended!" << endll;
}

void ActionEngine::start()
{
    LOG(LOG_INFO) << std::setw(12) << "engine:" << std::setw(18) << "[ActionEngine]" << " started!" << endll;
    is_alive_ = true;
    td_ = std::move(std::thread(&ActionEngine::run, this));
}

void ActionEngine::stop()
{
    is_alive_ = false;
}

void ActionEngine::set_params(const std::vector< std::map<robot::robot_motion, robot::robot_pose> > &poses, const std::vector<int> &pos_times)
{
    std::lock_guard<std::mutex> lk(param_mtx_);
    if(!poses_.empty()) return;
    poses_ = poses;
    pos_times_ = pos_times;
}

void ActionEngine::run()
{
    std::vector< std::map<robot::robot_motion, robot::robot_pose> > poses_temp;
    std::vector<int> pos_times_temp;
    while(is_alive_)
    {
        param_mtx_.lock();
        poses_temp.clear();
        poses_temp = poses_;
        pos_times_temp.clear();
        pos_times_temp = pos_times_;
        param_mtx_.unlock();
        if(!poses_temp.empty())
        {
            MADT->run_action_ = true;
            LOG(LOG_WARN)<<"ready to act"<<endll;
            if(WE->get_walk_state() != WALK_STOP)
            {
                WE->set_walk_state(WALK_TO_ACT);
                while (WE->get_walk_state() != WALK_STOP && is_alive_)
                {
                    usleep(500);
                }
            }
            LOG(LOG_WARN)<<"action"<<endll;
            if(!is_alive_) break;
            int act_time;
            std::map<int, float> one_pos_deg;

            if (!poses_temp.empty() && !pos_times_temp.empty())
            {
                in_action_ = true;
                std::map<std::string, float> jdegs;
                std::map<robot::robot_motion, robot::robot_pose> pos1, pos2;
                robot_math::TransformMatrix body_mat, leftfoot_mat, rightfoot_mat;
                Eigen::Vector3d lefthand, righthand;

                act_time = pos_times_temp[0];
                pos1 = poses_temp[0];

                body_mat = robot::ROBOT->get_body_mat_from_pose(pos1[robot::MOTION_BODY]);
                leftfoot_mat = robot::ROBOT->get_foot_mat_from_pose(pos1[robot::MOTION_LEFT_FOOT], true);
                rightfoot_mat = robot::ROBOT->get_foot_mat_from_pose(pos1[robot::MOTION_RIGHT_FOOT], false);
                righthand[0] = pos1[robot::MOTION_RIGHT_HAND].x;
                righthand[2] = pos1[robot::MOTION_RIGHT_HAND].z;
                lefthand[0] = pos1[robot::MOTION_LEFT_HAND].x;
                lefthand[2] = pos1[robot::MOTION_LEFT_HAND].z;

                if (get_degs(body_mat, leftfoot_mat, rightfoot_mat, lefthand, righthand, one_pos_deg))
                {
                    int odc=0;
                    for(auto od:one_pos_deg)
                    {
                        if(fabs(ROBOT->get_joint(od.first)->get_deg()-od.second)<1E-2)
                            ++odc;
                    }
                    if(odc!=one_pos_deg.size())
                        if (!set_joints(one_pos_deg, act_time))
                            break;
                }

                if (poses_temp.size() >= 2)
                {
                    for (int i = 1; i < poses_temp.size(); i++)
                    {
                        pos1 = poses_temp[i - 1];
                        pos2 = poses_temp[i];
                        std::vector< std::map<robot::robot_motion, robot::robot_pose> > act_poses;
                        act_poses = get_poses(pos1, pos2, pos_times_temp[i]);
                        if(i==poses_temp.size()-1)
                            act_poses.push_back(pos2);

                        for (auto act_pos : act_poses)
                        {
                            body_mat = robot::ROBOT->get_body_mat_from_pose(act_pos[robot::MOTION_BODY]);
                            leftfoot_mat = robot::ROBOT->get_foot_mat_from_pose(act_pos[robot::MOTION_LEFT_FOOT], true);
                            rightfoot_mat = robot::ROBOT->get_foot_mat_from_pose(act_pos[robot::MOTION_RIGHT_FOOT], false);
                            righthand[0] = act_pos[robot::MOTION_RIGHT_HAND].x;
                            righthand[2] = act_pos[robot::MOTION_RIGHT_HAND].z;
                            lefthand[0] = act_pos[robot::MOTION_LEFT_HAND].x;
                            lefthand[2] = act_pos[robot::MOTION_LEFT_HAND].z;

                            if (get_degs(body_mat, leftfoot_mat, rightfoot_mat, lefthand, righthand, one_pos_deg))
                            {
                                if (!set_joints(one_pos_deg, 1))
                                {
                                    break;
                                }
                            }
                            else
                            {
                                LOG(LOG_WARN) << "ik error" << endll;
                            }
                        }
                    }
                }
            }
            LOG(LOG_WARN)<<"action end"<<endll;
            param_mtx_.lock();
            poses_.clear();
            pos_times_.clear();
            param_mtx_.unlock();
            MADT->run_action_ = false;
            in_action_ = false;
        }
        usleep(1000);
    }
}

std::vector< std::map<robot::robot_motion, robot::robot_pose> > ActionEngine::get_poses(std::map<robot::robot_motion, robot::robot_pose> &pos1,
            std::map<robot::robot_motion, robot::robot_pose> &pos2, int act_time)
{
    Eigen::Vector3d poseb1(pos1[robot::MOTION_BODY].x, pos1[robot::MOTION_BODY].y, pos1[robot::MOTION_BODY].z);
    Eigen::Vector3d poseb2(pos2[robot::MOTION_BODY].x, pos2[robot::MOTION_BODY].y, pos2[robot::MOTION_BODY].z);
    Eigen::Vector3d posel1(pos1[robot::MOTION_LEFT_FOOT].x, pos1[robot::MOTION_LEFT_FOOT].y, pos1[robot::MOTION_LEFT_FOOT].z);
    Eigen::Vector3d posel2(pos2[robot::MOTION_LEFT_FOOT].x, pos2[robot::MOTION_LEFT_FOOT].y, pos2[robot::MOTION_LEFT_FOOT].z);
    Eigen::Vector3d poser1(pos1[robot::MOTION_RIGHT_FOOT].x, pos1[robot::MOTION_RIGHT_FOOT].y, pos1[robot::MOTION_RIGHT_FOOT].z);
    Eigen::Vector3d poser2(pos2[robot::MOTION_RIGHT_FOOT].x, pos2[robot::MOTION_RIGHT_FOOT].y, pos2[robot::MOTION_RIGHT_FOOT].z);
    Eigen::Vector3d pposeb1(pos1[robot::MOTION_BODY].pitch, pos1[robot::MOTION_BODY].roll, pos1[robot::MOTION_BODY].yaw);
    Eigen::Vector3d pposeb2(pos2[robot::MOTION_BODY].pitch, pos2[robot::MOTION_BODY].roll, pos2[robot::MOTION_BODY].yaw);
    Eigen::Vector3d pposel1(pos1[robot::MOTION_LEFT_FOOT].pitch, pos1[robot::MOTION_LEFT_FOOT].roll, pos1[robot::MOTION_LEFT_FOOT].yaw);
    Eigen::Vector3d pposel2(pos2[robot::MOTION_LEFT_FOOT].pitch, pos2[robot::MOTION_LEFT_FOOT].roll, pos2[robot::MOTION_LEFT_FOOT].yaw);
    Eigen::Vector3d pposer1(pos1[robot::MOTION_RIGHT_FOOT].pitch, pos1[robot::MOTION_RIGHT_FOOT].roll, pos1[robot::MOTION_RIGHT_FOOT].yaw);
    Eigen::Vector3d pposer2(pos2[robot::MOTION_RIGHT_FOOT].pitch, pos2[robot::MOTION_RIGHT_FOOT].roll, pos2[robot::MOTION_RIGHT_FOOT].yaw);

    Eigen::Vector3d poselh1(pos1[robot::MOTION_LEFT_HAND].x, pos1[robot::MOTION_LEFT_HAND].y, pos1[robot::MOTION_LEFT_HAND].z);
    Eigen::Vector3d poselh2(pos2[robot::MOTION_LEFT_HAND].x, pos2[robot::MOTION_LEFT_HAND].y, pos2[robot::MOTION_LEFT_HAND].z);
    Eigen::Vector3d poserh1(pos1[robot::MOTION_RIGHT_HAND].x, pos1[robot::MOTION_RIGHT_HAND].y, pos1[robot::MOTION_RIGHT_HAND].z);
    Eigen::Vector3d poserh2(pos2[robot::MOTION_RIGHT_HAND].x, pos2[robot::MOTION_RIGHT_HAND].y, pos2[robot::MOTION_RIGHT_HAND].z);

    Eigen::Vector3d dposeb = poseb2 - poseb1;
    Eigen::Vector3d dposel = posel2 - posel1;
    Eigen::Vector3d dposer = poser2 - poser1;
    Eigen::Vector3d dposelh = poselh2 - poselh1;
    Eigen::Vector3d dposerh = poserh2 - poserh1;

    Eigen::Vector3d dpposeb = pposeb2 - pposeb1;
    Eigen::Vector3d dpposel = pposel2 - pposel1;
    Eigen::Vector3d dpposer = pposer2 - pposer1;
    int count;
    count = act_time;
    double dbx = dposeb.x() / count, dby = dposeb.y() / count, dbz = dposeb.z() / count;
    double dbpi = dpposeb.x() / count, dbro = dpposeb.y() / count, dbya = dpposeb.z() / count;
    double dlx = dposel.x() / count, dly = dposel.y() / count, dlz = dposel.z() / count;
    double dlpi = dpposel.x() / count, dlro = dpposel.y() / count, dlya = dpposel.z() / count;
    double drx = dposer.x() / count, dry = dposer.y() / count, drz = dposer.z() / count;
    double drpi = dpposer.x() / count, drro = dpposer.y() / count, drya = dpposer.z() / count;

    double dlhx = dposelh.x() / count, dlhz = dposelh.z() / count;
    double drhx = dposerh.x() / count, drhz = dposerh.z() / count;
    std::vector< std::map<robot::robot_motion, robot::robot_pose> > act_poses;
    for (int i = 0; i < count; i++)
    {
        std::map<robot::robot_motion, robot::robot_pose> temp_pose_map;
        robot::robot_pose temp_pose;
        temp_pose.x = poselh1.x() + i * dlhx;
        temp_pose.z = poselh1.z() + i * dlhz;
        temp_pose_map[robot::MOTION_LEFT_HAND] = temp_pose;
        temp_pose.x = poserh1.x() + i * drhx;
        temp_pose.z = poserh1.z() + i * drhz;
        temp_pose_map[robot::MOTION_RIGHT_HAND] = temp_pose;
        temp_pose.x = poseb1.x() + i * dbx;
        temp_pose.y = poseb1.y() + i * dby;
        temp_pose.z = poseb1.z() + i * dbz;
        temp_pose.pitch = pposeb1.x() + i * dbpi;
        temp_pose.roll = pposeb1.y() + i * dbro;
        temp_pose.yaw = pposeb1.z() + i * dbya;
        temp_pose_map[robot::MOTION_BODY] = temp_pose;
        temp_pose.x = posel1.x() + i * dlx;
        temp_pose.y = posel1.y() + i * dly;
        temp_pose.z = posel1.z() + i * dlz;
        temp_pose.pitch = pposel1.x() + i * dlpi;
        temp_pose.roll = pposel1.y() + i * dlro;
        temp_pose.yaw = pposel1.z() + i * dlya;
        temp_pose_map[robot::MOTION_LEFT_FOOT] = temp_pose;
        temp_pose.x = poser1.x() + i * drx;
        temp_pose.y = poser1.y() + i * dry;
        temp_pose.z = poser1.z() + i * drz;
        temp_pose.pitch = pposer1.x() + i * drpi;
        temp_pose.roll = pposer1.y() + i * drro;
        temp_pose.yaw = pposer1.z() + i * drya;
        temp_pose_map[robot::MOTION_RIGHT_FOOT] = temp_pose;
        act_poses.push_back(temp_pose_map);
    }
    return act_poses;
}

bool ActionEngine::get_degs(const robot_math::TransformMatrix &body_mat, const robot_math::TransformMatrix &leftfoot_mat,
                const robot_math::TransformMatrix &rightfoot_mat, const Eigen::Vector3d &lefthand, const Eigen::Vector3d &righthand,
                std::map<int, float> &degs_map)
{
    std::vector<double> degs;
    degs_map.clear();

    if (robot::ROBOT->leg_inverse_kinematics(body_mat, leftfoot_mat, degs, true))
    {
        degs_map[robot::ROBOT->get_joint("jlhip3")->jid_] = robot_math::rad2deg(degs[0]);
        degs_map[robot::ROBOT->get_joint("jlhip2")->jid_] = robot_math::rad2deg(degs[1]);
        degs_map[robot::ROBOT->get_joint("jlhip1")->jid_] = robot_math::rad2deg(degs[2]);
        degs_map[robot::ROBOT->get_joint("jlknee")->jid_] = robot_math::rad2deg(degs[3]);
        degs_map[robot::ROBOT->get_joint("jlankle2")->jid_] = robot_math::rad2deg(degs[4]);
        degs_map[robot::ROBOT->get_joint("jlankle1")->jid_] = robot_math::rad2deg(degs[5]);
    }
    else
    {
        LOG(LOG_ERROR) <<"IK ERROR!"<<endll;
        return false;
    }

    if (robot::ROBOT->leg_inverse_kinematics(body_mat, rightfoot_mat, degs, false))
    {
        degs_map[robot::ROBOT->get_joint("jrhip3")->jid_] = robot_math::rad2deg(degs[0]);
        degs_map[robot::ROBOT->get_joint("jrhip2")->jid_] = robot_math::rad2deg(degs[1]);
        degs_map[robot::ROBOT->get_joint("jrhip1")->jid_] = robot_math::rad2deg(degs[2]);
        degs_map[robot::ROBOT->get_joint("jrknee")->jid_] = robot_math::rad2deg(degs[3]);
        degs_map[robot::ROBOT->get_joint("jrankle2")->jid_] = robot_math::rad2deg(degs[4]);
        degs_map[robot::ROBOT->get_joint("jrankle1")->jid_] = robot_math::rad2deg(degs[5]);
    }
    else
    {
        LOG(LOG_ERROR) <<"IK ERROR!"<<endll;
        return false;
    }

    if (robot::ROBOT->arm_inverse_kinematics(lefthand, degs))
    {
        degs_map[robot::ROBOT->get_joint("jlshoulder1")->jid_] = robot_math::rad2deg(degs[0]);
        degs_map[robot::ROBOT->get_joint("jlelbow")->jid_] = -robot_math::rad2deg(degs[2]);
    }
    else
    {
        LOG(LOG_ERROR) <<"IK ERROR!"<<endll;
        return false;
    }

    if (robot::ROBOT->arm_inverse_kinematics(righthand, degs))
    {
        degs_map[robot::ROBOT->get_joint("jrshoulder1")->jid_] = robot_math::rad2deg(degs[0]);
        degs_map[robot::ROBOT->get_joint("jrelbow")->jid_] = robot_math::rad2deg(degs[2]);
    }
    else
    {
        LOG(LOG_ERROR) <<"IK ERROR!"<<endll;
        return false;
    }

    return true;
}

bool ActionEngine::set_joints(std::map<int, float> &jdegs, int act_time)
{
    std::map<int, float> latest_deg, diff, jdmap;
    latest_deg = MADT->get_body_degs();
    for (auto &jd : latest_deg)
    {
        diff[jd.first] = jdegs[jd.first] - latest_deg[jd.first];
    }

    for (int i = 1; i <= act_time; i++)
    {
        jdmap.clear();

        for (auto &jd : diff)
        {
            jdmap[jd.first] = latest_deg[jd.first] + i * jd.second / (float)act_time;
        }
        while (!MADT->body_empty())
        {
            usleep(1000);
        }

        if (!MADT->add_body_degs(jdmap))
        {
            return false;
        }
    }

    return true;
}
}