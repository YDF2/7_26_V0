#ifndef __HEAR_HPP
#define __HEAR_HPP

#include <boost/asio.hpp>
#include <thread>
#include "sensor.hpp"
#include "udp_data/CommData.h"

class Hear: public Sensor
{
public:
    Hear();
    ~Hear();

    bool start();
    void stop();

    inline player_info info() const
    {
        return pkt_.info;
    }

private:
    void receive();
    boost::asio::ip::udp::socket socket_;
    boost::asio::ip::udp::endpoint point_;
    std::thread td_;
    comm_packet pkt_;
};

#endif
