#ifndef __GAME_CTRL_HPP
#define __GAME_CTRL_HPP

#include <boost/asio.hpp>
#include <memory>
#include <thread>
#include "udp_data/RoboCupGameControlData.h"
#include "sensor.hpp"

class GameCtrl: public Sensor
{
public:
    // Game states (mapping to STATE_* defines)
    enum RobocupGameState
    {
        GS_INITIAL = STATE_INITIAL,     // 0
        GS_READY = STATE_READY,         // 1
        GS_SET = STATE_SET,             // 2
        GS_PLAY = STATE_PLAYING,        // 3
        GS_FINISH = STATE_FINISHED      // 4
    };

    // Game phases (mapping to GAME_PHASE_* defines)
    enum RobocupGamePhase
    {
        GP_NORMAL = GAME_PHASE_NORMAL,              // 0
        GP_PENALTY_SHOOT_OUT = GAME_PHASE_PENALTY_SHOOT_OUT, // 1
        GP_EXTRA_TIME = GAME_PHASE_EXTRA_TIME,      // 2
        GP_TIMEOUT = GAME_PHASE_TIMEOUT             // 3
    };

    // Set play types (mapping to SET_PLAY_* defines)
    enum RobocupSetPlay
    {
        SP_NONE = SET_PLAY_NONE,                    // 0
        SP_DIRECT_FREE_KICK = SET_PLAY_DIRECT_FREE_KICK,   // 1
        SP_INDIRECT_FREE_KICK = SET_PLAY_INDIRECT_FREE_KICK, // 2
        SP_PENALTY_KICK = SET_PLAY_PENALTY_KICK,    // 3
        SP_THROW_IN = SET_PLAY_THROW_IN,            // 4
        SP_GOAL_KICK = SET_PLAY_GOAL_KICK,          // 5
        SP_CORNER_KICK = SET_PLAY_CORNER_KICK       // 6
    };

    // Penalty states (mapping to PENALTY_* defines)
    enum RobocupPlayerState
    {
        P_NONE = PENALTY_NONE,                              // 0
        P_ILLEGAL_POSITIONING = PENALTY_ILLEGAL_POSITIONING, // 1
        P_MOTION_IN_SET = PENALTY_MOTION_IN_SET,            // 2
        P_LOCAL_GAME_STUCK = PENALTY_LOCAL_GAME_STUCK,      // 3
        P_INCAPABLE_ROBOT = PENALTY_INCAPABLE_ROBOT,        // 4 - robot incapable (pickup)
        P_PICK_UP = PENALTY_PICK_UP,                        // 5 - pickup penalty
        P_BALL_HOLDING = PENALTY_BALL_HOLDING,              // 6
        P_LEAVING_THE_FIELD = PENALTY_LEAVING_THE_FIELD,    // 7
        P_PLAYING_WITH_ARMS_HANDS = PENALTY_PLAYING_WITH_ARMS_HANDS, // 8
        P_PUSHING = PENALTY_PUSHING,                        // 9
        P_SENT_OFF = PENALTY_SENT_OFF,                      // 10
        P_SUBSTITUTE = PENALTY_SUBSTITUTE                   // 11
    };

    // Free kick stages (kept for internal use)
    enum FreeKickStage
    {
        FK_ADJUST = 0,
        FK_EXECUTE
    };

    GameCtrl();
    ~GameCtrl();

    bool start();
    void stop();

    inline RoboCupGameControlData data() const
    {
        return data_;
    }
private:
    void receive();

    enum {gc_data_size = sizeof(RoboCupGameControlData)};
    RoboCupGameControlData data_;
    std::thread td_;
    boost::asio::ip::udp::socket recv_socket_;
    boost::asio::ip::udp::endpoint recv_point_;
};

#endif