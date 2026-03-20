#ifndef __GCRET_TASK_HPP
#define __GCRET_TASK_HPP

#include <boost/asio.hpp>
#include <exception>
#include "task.hpp"
#include "udp_data/RoboCupGameControlData.h"
#include "configuration.hpp"

class GcretTask: public Task
{
public:
    GcretTask(): Task("gcret")
    {
    }

    bool perform()
    {
        RoboCupGameControlReturnData ret_data;
        ret_data.message = GAMECONTROLLER_RETURN_MSG_ALIVE;
        ret_data.player = static_cast<uint8_t>(CONF->id());
        ret_data.team = CONF->get_config_value<uint8_t>("team_number");
        /*boost::asio::ip::address::from_string(CONF->get_config_value<string>("net.udp.addr"))*/
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
