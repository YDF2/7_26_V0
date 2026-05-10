#ifndef __GCRET_TASK_HPP
#define __GCRET_TASK_HPP

#include <boost/asio.hpp>
#include <exception>
#include <cmath>
#include "task.hpp"
#include "udp_data/RoboCupGameControlData.h"
#include "configuration.hpp"
#include "core/worldmodel.hpp"

class GcretTask: public Task
{
public:
    GcretTask(): Task("gcret")
    {
    }

    bool perform()
    {
        RoboCupGameControlReturnData ret_data;

        // Fill in the new structure fields
        ret_data.playerNum = static_cast<uint8_t>(CONF->id());
        ret_data.teamNum = CONF->get_config_value<uint8_t>("team_number");

        // Get fallen status from IMU/fall detection
        int fall_dir = WM->fall_data();
        ret_data.fallen = (fall_dir != 0) ? 1 : 0;  // 1 if fallen, 0 if can play

        // Get pose (position and orientation) from WorldModel
        self_block self = WM->self();
        // Convert meters to millimeters (pose units are mm)
        ret_data.pose[0] = static_cast<float>(self.global.x() * 1000.0f);  // x in mm
        ret_data.pose[1] = static_cast<float>(self.global.y() * 1000.0f);  // y in mm
        ret_data.pose[2] = static_cast<float>(self.dir * M_PI / 180.0f);   // theta in radians

        // Get ball information
        ball_block ball = WM->ball();
        if (ball.can_see)
        {
            ret_data.ballAge = 0.0f;  // Just saw the ball
            // Ball position relative to robot (in mm)
            ret_data.ball[0] = static_cast<float>(ball.self.x() * 1000.0f);  // x in mm
            ret_data.ball[1] = static_cast<float>(ball.self.y() * 1000.0f);  // y in mm
        }
        else
        {
            ret_data.ballAge = -1.0f;  // Haven't seen ball
            ret_data.ball[0] = 0.0f;
            ret_data.ball[1] = 0.0f;
        }

        try
        {
            boost::asio::io_service gcret_service;
            boost::asio::ip::udp::socket send_socket(gcret_service, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0));
            send_socket.set_option(boost::asio::socket_base::broadcast(true));
            boost::asio::ip::udp::endpoint send_point(boost::asio::ip::address_v4::broadcast(), GAMECONTROLLER_RETURN_PORT);
            send_socket.async_send_to(boost::asio::buffer((char *)(&ret_data), sizeof(RoboCupGameControlReturnData)), send_point,
            [this](boost::system::error_code ec, std::size_t bytes_sent) {});
            return true;
        }
        catch (std::exception &e)
        {
            LOG(LOG_WARN) << e.what() << endll;
            return false;
        }
    }
};

#endif