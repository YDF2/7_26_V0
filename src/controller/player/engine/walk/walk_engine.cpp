#include "walk_engine.hpp"
#include "CubicSpline.hpp"
#include "SmoothSpline.hpp"
#include "Polynom.hpp"
#include "robot/robot.hpp"
#include "math/math.hpp"
#include "configuration.hpp"
#include <cmath>
#include <fstream>
#include "core/adapter.hpp"
#include "sensor/motor.hpp"
#include "core/worldmodel.hpp"
#include "logger.hpp"

namespace motion
{
    using namespace Eigen;
    using namespace robot;
    using namespace robot_math;
    using namespace std;

    WalkEngine::WalkEngine()
    {
        string part = CONF->player() + ".walk";
        nav_coef_ = CONF->get_config_vector<double, 2>(part + ".nav");
        XOffset_ = CONF->get_config_value<double>(part + ".XOffset");
        YOffset_ = CONF->get_config_value<double>(part + ".YOffset");
        DOffset_ = CONF->get_config_value<double>(part + ".DOffset");

        xrange_ = CONF->get_config_vector<double, 2>(part + ".x");
        yrange_ = CONF->get_config_vector<double, 2>(part + ".y");
        drange_ = CONF->get_config_vector<double, 2>(part + ".dir");
        /**
         * Model leg typical length between
         * each rotation axis
         */
        params_.distHipToKnee = ROBOT->A();
        params_.distKneeToAnkle = ROBOT->B();
        params_.distAnkleToGround = ROBOT->C();
        /**
         * Distance between the two feet in lateral
         * axis while in zero position
         */
        params_.distFeetLateral = ROBOT->D();
        /**
         * Complete (two legs) walk cycle frequency
         * in Hertz
         */
        params_.freq = CONF->get_config_value<double>(part + ".freq");
        /**
         * Global gain multiplying all time
         * dependant movement between 0 and 1.
         * Control walk enabled/disabled smoothing.
         * 0 is walk disabled.
         * 1 is walk fully enabled
         */
        params_.enabledGain = 1.0;
        /**
         * Length of double support phase
         * in phase time
         * (between 0 and 1)
         * 0 is null double support and full single support
         * 1 is full double support and null single support
         */
        params_.supportPhaseRatio = CONF->get_config_value<double>(part + ".doubleSupportRatio");
        /**
         * Lateral offset on default foot
         * position in meters (foot lateral distance)
         * 0 is default
         * > 0 is both feet external offset
         */
        params_.footYOffset = CONF->get_config_value<double>(part + ".footYOffset");
        ;
        /**
         * Forward length of each foot step
         * in meters
         * >0 goes forward
         * <0 goes backward
         * (dynamic parameter)
         */
        params_.stepGain = 0.0;
        /**
         * Vertical rise height of each foot
         * in meters (positive)
         */
        params_.riseGain = CONF->get_config_value<double>(part + ".rise");
        ;
        /**
         * Angular yaw rotation of each
         * foot for each step in radian.
         * 0 does not turn
         * >0 turns left
         * <0 turns right
         * (dynamic parameter)
         */
        params_.turnGain = 0.0;
        /**
         * Lateral length of each foot step
         * in meters.
         * >0 goes left
         * <0 goes right
         * (dynamic parameter)
         */
        params_.lateralGain = 0.0;
        /**
         * Vertical foot offset from trunk
         * in meters (positive)
         * 0 is in init position
         * > 0 set the robot lower to the ground
         */
        params_.trunkZOffset = CONF->get_config_value<double>(part + ".trunkZOffset");
        ;
        /**
         * Lateral trunk oscillation amplitude
         * in meters (positive)
         */
        params_.swingGain = CONF->get_config_value<double>(part + ".swingGain");
        ;
        /**
         * Lateral angular oscillation amplitude
         * of swing trunkRoll in radian
         */
        params_.swingRollGain = 0.0;
        /**
         * Phase shift of lateral trunk oscillation
         * between 0 and 1
         */
        params_.swingPhase = CONF->get_config_value<double>(part + ".swingPhase");
        ;
        /**
         * Foot X-Z spline velocities
         * at ground take off and ground landing.
         * Step stands for X and rise stands for Z
         * velocities.
         * Typical values ranges within 0 and 5.
         * >0 for DownVel is having the foot touching the
         * ground with backward velocity.
         * >0 for UpVel is having the foot going back
         * forward with non perpendicular tangent.
         */
        params_.stepUpVel = 4.0;
        params_.stepDownVel = 4.0;
        params_.riseUpVel = 4.0;
        params_.riseDownVel = 4.0;
        /**
         * Time length in phase time
         * where swing lateral oscillation
         * remains on the same side
         * between 0 and 0.5
         */
        params_.swingPause = 0.0;
        /**
         * Swing lateral spline velocity (positive).
         * Control the "smoothness" of swing trajectory.
         * Typical values are between 0 and 5.
         */
        params_.swingVel = 4.0;
        /**
         * Forward trunk-foot offset
         * with respect to foot in meters
         * >0 moves the trunk forward
         * <0 moves the trunk backward
         */
        params_.trunkXOffset = CONF->get_config_value<double>(part + ".trunkXOffset");
        ;
        /**
         * Lateral trunk-foot offset
         * with respect to foot in meters
         * >0 moves the trunk on the left
         * <0 moves the trunk on the right
         */
        params_.trunkYOffset = CONF->get_config_value<double>(part + ".trunkYOffset");
        ;
        /**
         * Trunk angular rotation
         * around Y in radian
         * >0 bends the trunk forward
         * <0 bends the trunk backward
         */
        params_.trunkPitch = deg2rad(CONF->get_config_value<double>(part + ".trunkPitch"));
        /**
         * Trunk angular rotation
         * around X in radian
         * >0 bends the trunk on the right
         * <0 bends the trunk on the left
         */
        params_.trunkRoll = 0.0;
        /**
         * Add extra offset on X, Y and Z
         * direction on left and right feet
         * in meters
         * (Can be used for example to implement
         * dynamic kick)
         */
        params_.extraLeftX = 0.0;
        params_.extraLeftY = 0.0;
        params_.extraLeftZ = 0.0;
        params_.extraRightX = 0.0;
        params_.extraRightY = 0.0;
        params_.extraRightZ = 0.0;
        /**
         * Add extra angular offset on
         * Yaw, Pitch and Roll rotation of
         * left and right foot in radians
         */
        params_.extraLeftYaw = 0.0;
        params_.extraLeftPitch = 0.0;
        params_.extraLeftRoll = 0.0;
        params_.extraRightYaw = 0.0;
        params_.extraRightPitch = 0.0;
        params_.extraRightRoll = 0.0;

        // The walk is started while walking on place
        params_.enabledGain = 0.0;
        params_.stepGain = 0.0;
        params_.lateralGain = 0.0;
        params_.turnGain = 0.0;

        phase_ = 0.0;
        time_ = 0.0;
        engine_frequency_ = 1000.0 / CONF->get_config_value<int>("hardware.motor.period");
        time_length_ = 1.0 / params_.freq;

        walk_state_ = WALK_STOP;
        last_walk_state_ = WALK_STOP;
    }

    void WalkEngine::updata(const pub_ptr &pub, const int &type)
    {
        if (type == Sensor::SENSOR_IMU)
        {
            std::shared_ptr<Imu> sptr = std::dynamic_pointer_cast<Imu>(pub);
            imu_mtx_.lock();
            imu_data_ = sptr->data();
            // LOG(LOG_INFO) << "imu:" << imu_data_.yaw << endl;
            imu_mtx_.unlock();
            return;
        }

        if (type == Sensor::SENSOR_MOTOR)
        {
            std::shared_ptr<Motor> sptr = std::dynamic_pointer_cast<Motor>(pub);
            dxl_mtx_.lock();

            dxl_mtx_.unlock();
            return;
        }
    }

    WalkEngine::~WalkEngine()
    {
        if (td_.joinable())
        {
            td_.join();
        }
        LOG(LOG_INFO) << std::setw(12) << "engine:" << std::setw(18) << "[WalkEngine]"
                      << " ended!" << endll;
    }

    void WalkEngine::start()
    {
        is_alive_ = true;
        td_ = std::move(std::thread(&WalkEngine::run, this));
        LOG(LOG_INFO) << std::setw(12) << "engine:" << std::setw(18) << "[WalkEngine]"
                      << " started!" << endll;
    }

    void WalkEngine::set_params(float x, float y, float d, bool enable)
    {
        if (MADT->run_action_)
            return;
        para_mutex_.lock();
        params_.stepGain = x;
        params_.lateralGain = y; //横向步长
        params_.turnGain = d;
        params_.enabledGain = enable ? 1.0 : 0.0;
        walk_state_ = enable ? WALK_NORMAL : WALK_STOP;
        bound(xrange_[0], xrange_[1], params_.stepGain);
        bound(yrange_[0], yrange_[1], params_.lateralGain);
        bound(drange_[0], drange_[1], params_.turnGain);
        params_.stepGain += XOffset_;
        params_.lateralGain += YOffset_;
        params_.turnGain = deg2rad(params_.turnGain) + deg2rad(DOffset_); //注意此处为-号，change for + ,in order to be the same with direct params
                                                                          //that + indicates left and - indicates right
        para_mutex_.unlock();
    }

    void WalkEngine::run_walk(const Rhoban::IKWalkParameters &params, double timeLength, double &phase, double &time)
    {
        // LOG(LOG_WARN)<<"run_walk:"<< walk_state_<<endll; //新加
        struct Rhoban::IKWalkOutputs outputs;
        std::map<int, float> jdegs;
        float init_dir = WM->self().dir;
        for (double t = 0.0; t <= timeLength; t += 1.0 / engine_frequency_)
        {
            time += 1.0 / engine_frequency_;
            bool success = Rhoban::IKWalk::walk(params, 1.0 / engine_frequency_, phase, outputs);
            if (!success)
            {
                LOG(LOG_ERROR) << time << " Inverse Kinematics error. Position not reachable." << endll;
            }
            else
            {
                jdegs[ROBOT->get_joint("jlhip3")->jid_] = rad2deg(outputs.left_hip_yaw);
                jdegs[ROBOT->get_joint("jlhip2")->jid_] = rad2deg(outputs.left_hip_roll);
                jdegs[ROBOT->get_joint("jlhip1")->jid_] = rad2deg(outputs.left_hip_pitch);
                jdegs[ROBOT->get_joint("jlknee")->jid_] = rad2deg(outputs.left_knee);
                jdegs[ROBOT->get_joint("jlankle2")->jid_] = rad2deg(outputs.left_ankle_pitch);
                jdegs[ROBOT->get_joint("jlankle1")->jid_] = rad2deg(outputs.left_ankle_roll);

                jdegs[ROBOT->get_joint("jrhip3")->jid_] = rad2deg(outputs.right_hip_yaw);
                jdegs[ROBOT->get_joint("jrhip2")->jid_] = rad2deg(outputs.right_hip_roll);
                jdegs[ROBOT->get_joint("jrhip1")->jid_] = rad2deg(outputs.right_hip_pitch);
                jdegs[ROBOT->get_joint("jrknee")->jid_] = rad2deg(outputs.right_knee);
                jdegs[ROBOT->get_joint("jrankle2")->jid_] = rad2deg(outputs.right_ankle_pitch);
                jdegs[ROBOT->get_joint("jrankle1")->jid_] = rad2deg(outputs.right_ankle_roll);

                jdegs[ROBOT->get_joint("jlshoulder1")->jid_] = 0;
                jdegs[ROBOT->get_joint("jlelbow")->jid_] = -170;
                jdegs[ROBOT->get_joint("jrshoulder1")->jid_] = 0;
                jdegs[ROBOT->get_joint("jrelbow")->jid_] = 170;

                while (!MADT->body_empty())
                {
                    usleep(500);
                }
                if (!MADT->add_body_degs(jdegs))
                {
                    break;
                }
            }
        }
        float rate = timeLength / time_length_;

        self_block blk = WM->self();
        Vector2d currpos(blk.global.x(), blk.global.y()); //当前位置
        double dir = (init_dir + blk.dir) / 2;
        dir = normalize_deg(dir);
        //此处估计距离，修改walk2.conf中的nav来修正 nav_x越大，走的距离越短， nav_y暂时未改
        Vector2d temp = currpos + rotation_mat_2d(-dir) * Vector2d((params.stepGain - XOffset_) * rate * nav_coef_.x(),
                                                                   (params.lateralGain - YOffset_) * rate * nav_coef_.y());
        WM->set_my_pos(temp);
    }

    void WalkEngine::run()
    {
        const vector<string> walk_s{
            "WALK_STOP",
            "WALK_START",
            "WALK_END",
            "WALK_TO_ACT",
            "WALK_NORMAL"};
        Rhoban::IKWalkParameters tempParams;
        while (is_alive_)
        {
            para_mutex_.lock();
            tempParams = params_;
            para_mutex_.unlock();

            if (last_walk_state_ == WALK_STOP && walk_state_ == WALK_NORMAL && !MADT->run_action_)
                walk_state_ = WALK_START;
            if (last_walk_state_ == WALK_NORMAL && walk_state_ == WALK_STOP)
                walk_state_ = WALK_END;

            if (walk_state_ == WALK_STOP)
            {
                usleep(500);
                last_walk_state_ = WALK_STOP;
                continue;
            }
            // LOG(LOG_INFO)<< "isalive:" << walk_s[walk_state_]<<endll; //新加
            if (walk_state_ == WALK_TO_ACT)
            {
                tempParams.stepGain = 0.0;
                tempParams.lateralGain = 0.0;
                tempParams.turnGain = 0.0;
                double yoffset = 0.035;
                tempParams.footYOffset = yoffset;
                if (last_walk_state_ == WALK_NORMAL)
                {
                    tempParams.enabledGain = 1.0;
                    run_walk(tempParams, time_length_, phase_, time_);
                    tempParams.enabledGain = 0.0;
                    run_walk(tempParams, time_length_, phase_, time_);
                }
                tempParams.footYOffset = 0.04;
                walk_state_ = WALK_STOP;
                last_walk_state_ = WALK_STOP;
            }
            else
            {
                if (MADT->run_action_)
                {
                    usleep(500);
                    last_walk_state_ = WALK_STOP;
                    continue;
                }
                if (walk_state_ == WALK_START)
                {
                    LOG(LOG_INFO) << "walk start" << endll;
                    tempParams.stepGain = 0.0;
                    tempParams.lateralGain = 0.0;
                    tempParams.turnGain = 0.0;
                    double offset = tempParams.footYOffset;
                    tempParams.footYOffset = 0.025;
                    tempParams.riseGain = 0.02;
                    run_walk(tempParams, time_length_, phase_, time_);
                    tempParams.footYOffset = 0.0325;
                    tempParams.riseGain = 0.03;
                    run_walk(tempParams, time_length_, phase_, time_);
                    tempParams.footYOffset = offset;
                    tempParams.riseGain = 0.04;
                    run_walk(tempParams, 2 * time_length_, phase_, time_);
                    if (walk_state_ != WALK_TO_ACT)
                    {
                        walk_state_ = WALK_NORMAL;
                        last_walk_state_ = WALK_NORMAL;
                    }
                    else
                    {
                        walk_state_ = WALK_STOP;
                        last_walk_state_ = WALK_STOP;
                    }
                }
                else if (walk_state_ == WALK_END)
                {
                    LOG(LOG_INFO) << "walk end" << endll;
                    tempParams.stepGain = 0.0;
                    tempParams.lateralGain = 0.0;
                    tempParams.turnGain = 0.0;
                    tempParams.enabledGain = 1.0;
                    // run_walk(tempParams, 2*time_length_, phase_, time_);
                    // tempParams.footYOffset = 0.0325;
                    // run_walk(tempParams, time_length_, phase_, time_);
                    tempParams.footYOffset = 0.025;
                    run_walk(tempParams, 2 * time_length_, phase_, time_);
                    tempParams.enabledGain = 0.0;
                    run_walk(tempParams, 2 * time_length_, phase_, time_);
                    walk_state_ = WALK_STOP;
                    last_walk_state_ = WALK_STOP;
                }
                else if (walk_state_ == WALK_NORMAL)
                {
                    run_walk(tempParams, time_length_, phase_, time_);
                    last_walk_state_ = WALK_NORMAL;
                }
            }
            usleep(500);
        }
    }
}