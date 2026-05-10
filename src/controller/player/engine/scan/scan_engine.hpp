#ifndef __SCAN_ENGINE_HPP
#define __SCAN_ENGINE_HPP

#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "singleton.hpp"

namespace motion
{
    enum Head_State
    {
        HEAD_STATE_LOOKAT,
        HEAD_STATE_SEARCH_BALL,
        HEAD_STATE_SEARCH_POST,
        HEAD_STATE_TRACK_BALL
    };

    class ScanEngine: public Singleton<ScanEngine>
    {
    public:
        ScanEngine();
        ~ScanEngine();
        void start();
        void stop();
        void set_params(float yaw, float pitch, Head_State state);

        std::atomic_bool search_ball_end_, search_post_end_;
        float lost_yaw_, lost_pitch_;
        Eigen::Vector2f head_init_deg_;
        Eigen::Vector2f pitch_range_, yaw_range_;
        
    private:
        void run();
        std::thread td_;
        bool is_alive_;
        float yaw_, pitch_;

        std::atomic_int head_state_;
        const float search_post_div_ = 0.8;
        std::vector<Eigen::Vector2f> ball_search_table_;
        mutable std::mutex param_mtx_;
    };

    #define SE ScanEngine::instance()
}

#endif
