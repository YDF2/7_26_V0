#ifndef __PLAYER_HPP
#define __PLAYER_HPP

#include <list>
#include <atomic>
#include <thread>
#include <termios.h>
#include "timer.hpp"
#include "core/worldmodel.hpp"
#include "robot/robot.hpp"
#include "options/options.hpp"
#include "sensor/imu.hpp"
#include "sensor/motor.hpp"
#include "sensor/camera.hpp"
#include "sensor/button.hpp"
#include "vision/vision.hpp"
#include "common.hpp"
#include "task/task.hpp"
#include "fsm/fsm.hpp"

class Player: public Timer
{
public:
    Player();
    bool init();
    void stop();
    bool is_alive() const
    {
        return is_alive_;
    }

private:
    void run();
    void play_with_remote();
    std::list<task_ptr> play_with_gc();
    std::list<task_ptr> play_without_gc();
    std::list<task_ptr> play_manual();
    std::list<task_ptr> think();
    void start_manual_control();
    void stop_manual_control();
    void manual_control_loop();
    void apply_manual_key(char key);

    bool regist(); // 注册传感器
    void unregist();
    sensor_ptr get_sensor(const std::string &name);
private:
    unsigned long period_count_;
    std::map<std::string, sensor_ptr> sensors_;
    unsigned int btn_count_;
    std::string role_;
    unsigned int self_location_count_;
    bool played_;

    std::thread manual_thread_;
    std::atomic<bool> manual_thread_running_;
    std::atomic<float> manual_x_;
    std::atomic<float> manual_y_;
    std::atomic<float> manual_d_;
    std::atomic<bool> manual_enable_;
    std::atomic<long long> manual_last_input_ms_;
    std::atomic<bool> manual_kick_request_;
    std::atomic<long long> manual_last_kick_ms_;
    int manual_last_fall_dir_;
    struct termios terminal_old_;
    bool terminal_raw_;

    float manual_forward_step_;
    float manual_backward_step_;
    float manual_side_step_;
    float manual_turn_step_;
    int manual_input_timeout_ms_;
    int manual_kick_cooldown_ms_;
    bool manual_debug_log_;

    Eigen::Vector2d init_pos_, start_pos_, kickoff_pos_, pickup_pos_;
    FSM_Ptr fsm_;
};

#endif
