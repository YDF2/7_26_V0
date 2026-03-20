#ifndef __WALK_ENGINE_HPP
#define __WALK_ENGINE_HPP

#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <eigen3/Eigen/Dense>
#include "observer.hpp"
#include "sensor/imu.hpp"
#include "robot/robot_define.hpp"
#include "singleton.hpp"
#include "IKWalk.hpp"

namespace motion
{
    enum Walk_State
    {
        WALK_STOP,
        WALK_START,
        WALK_END,
        WALK_TO_ACT,
        WALK_NORMAL
    };

    class WalkEngine: public Subscriber, public Singleton<WalkEngine>
    {
    public:
        WalkEngine();
        ~WalkEngine();
        void start();
        void stop()
        {
            is_alive_ = false;
        }
        void set_params(float x, float y, float d, bool enable);
        void updata(const pub_ptr &pub, const int &type);
        void set_walk_state(Walk_State s)
        {
            walk_state_ = s;
        }

        int get_walk_state()
        {
            return walk_state_;
        }
        
    private:
        double engine_frequency_;

        double XOffset_, YOffset_, DOffset_;
        double phase_, time_;
        double time_length_;

        std::atomic_int walk_state_;
        int last_walk_state_;

        void run();

        void run_walk(const Rhoban::IKWalkParameters& params, double timeLength, double& phase, double& time);
        Rhoban::IKWalkParameters params_;
        std::thread td_;
        bool is_alive_;
        
        Eigen::Vector2d xrange_, yrange_, drange_, nav_coef_;
        Imu::imu_data imu_data_;
        mutable std::mutex para_mutex_, imu_mtx_, dxl_mtx_;
    };

#define WE WalkEngine::instance()
}

#endif
