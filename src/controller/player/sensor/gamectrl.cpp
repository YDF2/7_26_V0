#include "gamectrl.hpp"
#include "configuration.hpp"

using namespace std;
using boost::asio::ip::udp;

boost::asio::io_service gc_service;

GameCtrl::GameCtrl(): Sensor("gamectrl"),
    recv_socket_ (gc_service, udp::endpoint(udp::v4(), GAMECONTROLLER_DATA_PORT))
{

}

bool GameCtrl::start()
{
    is_open_ = true;
    is_alive_ = true;
    td_ = std::move(std::thread([this]()
    {
        this->receive();
        gc_service.run();
    }));
    return true;
}

void GameCtrl::receive()
{
    recv_socket_.async_receive_from(boost::asio::buffer((char *)&data_, gc_data_size), recv_point_,
                                    [this](boost::system::error_code ec, std::size_t bytes_recvd)
    {
        if (!ec && bytes_recvd > 0)
        {
            string recv_header;
            recv_header.append(data_.header, sizeof(RoboCupGameControlData::header));
            if (recv_header == GAMECONTROLLER_STRUCT_HEADER)
            {
                //LOG(LOG_INFO) << (int)data_.state<<endll;
                notify(SENSOR_GC);
            }
        }

        receive();
    });
}

void GameCtrl::stop()
{
    recv_socket_.close();
    gc_service.stop();
    is_alive_ = false;
    is_open_ = false;
}

GameCtrl::~GameCtrl()
{
    if (td_.joinable())
    {
        td_.join();
    }
}
