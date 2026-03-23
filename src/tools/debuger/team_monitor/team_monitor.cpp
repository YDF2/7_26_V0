#include "team_monitor.hpp"
#include "configuration.hpp"
#include "parser/parser.hpp"

using boost::asio::ip::udp;
using namespace std;

boost::asio::io_service udp_service;

TeamMonitor::TeamMonitor(): socket_(udp_service, udp::endpoint(udp::v4(), CONF->get_config_value<short>("net.udp")))
{
    parser::parse(CONF->field_file(), field_);
    field_.scale_field(0.5);
    setFixedSize(field_.field_length + 2 * field_.border_strip_width_min, field_.field_width + 2 * field_.border_strip_width_min);
    setStyleSheet("background:green");
    td_ = std::move(std::thread([this]()
    {
        this->receive();
        udp_service.run();
    }));
}

TeamMonitor::~TeamMonitor()
{
    if(td_.joinable())
    {
        td_.join();
    }
}

void TeamMonitor::receive()
{
    socket_.async_receive_from(boost::asio::buffer((char *)&pkt_, sizeof(comm_packet)), point_,
                               [this](boost::system::error_code ec, std::size_t bytes_recvd)
    {
        if (!ec && bytes_recvd > 0)
        {
            string recv_header;
            recv_header.append(pkt_.header, sizeof(comm_packet::header));
            if(recv_header==COMM_DATA_HEADER)
            {
                p_mutex_.lock();
                players_[pkt_.info.id] = pkt_.info;
                states_[pkt_.info.id] = pkt_.state;
                //cout<<pkt_.info.x<<'\t'<<pkt_.info.y<<'\t'<<pkt_.info.dir<<endll;
                p_mutex_.unlock();
                update();
            }
        }

        receive();
    });
}

void TeamMonitor::closeEvent(QCloseEvent *event)
{
    socket_.cancel();
    udp_service.stop();
    for(auto &m:state_monitors_)
        m.second->close();
    emit closed();
}

void TeamMonitor::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    painter.translate(field_.field_length / 2 + field_.border_strip_width_min, field_.field_width / 2 + field_.border_strip_width_min);
    painter.setPen(QPen(Qt::white, 2, Qt::SolidLine, Qt::FlatCap));
    painter.drawEllipse(-field_.center_circle_diameter / 2, -field_.center_circle_diameter / 2, field_.center_circle_diameter, field_.center_circle_diameter);
    painter.drawLine(0, -field_.field_width / 2, 0, field_.field_width / 2);
    painter.drawLine(-field_.field_length / 2, -field_.field_width / 2, field_.field_length / 2, -field_.field_width / 2);
    painter.drawLine(-field_.field_length / 2, -field_.field_width / 2, -field_.field_length / 2, field_.field_width / 2);
    painter.drawLine(-field_.field_length / 2, field_.field_width / 2, field_.field_length / 2, field_.field_width / 2);
    painter.drawLine(field_.field_length / 2, -field_.field_width / 2, field_.field_length / 2, field_.field_width / 2);
    painter.drawLine(-field_.field_length / 2, -field_.goal_area_width / 2, -(field_.field_length / 2 - field_.goal_area_length), -field_.goal_area_width / 2);
    painter.drawLine(-field_.field_length / 2, field_.goal_area_width / 2, -(field_.field_length / 2 - field_.goal_area_length), field_.goal_area_width / 2);
    painter.drawLine(-(field_.field_length / 2 - field_.goal_area_length), -field_.goal_area_width / 2, -(field_.field_length / 2 - field_.goal_area_length), field_.goal_area_width / 2);
    painter.drawLine(-(field_.field_length / 2 + field_.goal_depth), -field_.goal_width / 2, -field_.field_length / 2, -field_.goal_width / 2);
    painter.drawLine(-(field_.field_length / 2 + field_.goal_depth), field_.goal_width / 2, -field_.field_length / 2, field_.goal_width / 2);
    painter.drawLine(-(field_.field_length / 2 + field_.goal_depth), -field_.goal_width / 2, -(field_.field_length / 2 + field_.goal_depth), field_.goal_width / 2);
    painter.drawLine(field_.field_length / 2, -field_.goal_area_width / 2, (field_.field_length / 2 - field_.goal_area_length), -field_.goal_area_width / 2);
    painter.drawLine(field_.field_length / 2, field_.goal_area_width / 2, (field_.field_length / 2 - field_.goal_area_length), field_.goal_area_width / 2);
    painter.drawLine((field_.field_length / 2 - field_.goal_area_length), -field_.goal_area_width / 2, (field_.field_length / 2 - field_.goal_area_length), field_.goal_area_width / 2);
    painter.drawLine((field_.field_length / 2 + field_.goal_depth), -field_.goal_width / 2, field_.field_length / 2, -field_.goal_width / 2);
    painter.drawLine((field_.field_length / 2 + field_.goal_depth), field_.goal_width / 2, field_.field_length / 2, field_.goal_width / 2);
    painter.drawLine((field_.field_length / 2 + field_.goal_depth), -field_.goal_width / 2, (field_.field_length / 2 + field_.goal_depth), field_.goal_width / 2);

    painter.setPen(QPen(Qt::white, 2, Qt::SolidLine, Qt::FlatCap));
    painter.setBrush(QBrush(Qt::white, Qt::NoBrush));
    painter.drawRect(-(field_.field_length / 2 + field_.goal_depth), -field_.goal_width / 2, field_.goal_depth, field_.goal_width);
    painter.drawRect((field_.field_length / 2 + field_.goal_depth), -field_.goal_width / 2, -field_.goal_depth, field_.goal_width);
    painter.setPen(QPen(Qt::white, 1, Qt::SolidLine, Qt::FlatCap));
    painter.setBrush(QBrush(Qt::white, Qt::SolidPattern));
    painter.drawRect(-4, -4, 8, 8);
    painter.drawRect((field_.field_length / 2 - field_.penalty_mark_distance) - 4, -4, 8, 8);
    painter.drawRect(-(field_.field_length / 2 - field_.penalty_mark_distance) + 4, -4, 8, 8);

    int ballsize = 10;
    painter.setBrush(QBrush(Qt::black, Qt::SolidPattern));
    p_mutex_.lock();

    for (auto &p : players_)
    {
        if(p.second.my_kick)
        {
            painter.setPen(QPen(Qt::red, 2, Qt::SolidLine, Qt::FlatCap));
            painter.setBrush(QBrush(Qt::red, Qt::SolidPattern));
        }
        else
        {
            painter.setPen(QPen(Qt::black, 2, Qt::SolidLine, Qt::FlatCap));
            painter.setBrush(QBrush(Qt::black, Qt::SolidPattern));
        }
        painter.save();
        float s=field_.scale;
        painter.translate(p.second.x * 100*s, -p.second.y * 100*s);
        painter.drawEllipse(-ballsize / 2, -ballsize / 2, ballsize, ballsize);
        painter.drawText(-ballsize / 2, -ballsize / 2, QString::number(p.second.id));
        painter.rotate(-p.second.dir);
        painter.drawLine(0, 0, 2 * ballsize, 0);
        painter.restore();
        if(p.second.can_see)
        {
            painter.drawEllipse(p.second.ball_x * 100*s - ballsize / 2, -p.second.ball_y * 100*s - ballsize / 2, ballsize, ballsize);
            painter.drawText(p.second.ball_x*s * 100 - ballsize / 2, -p.second.ball_y*s * 100 - ballsize / 2, QString::number(p.second.id));
        }
        if(state_monitors_.find(p.second.id) == state_monitors_.end())
        {
            state_monitors_[p.second.id] = new StateMonitor(p.second.id);
            state_monitors_[p.second.id]->show();
        }
        else
        {
            state_monitors_[p.second.id]->update_state(states_[p.second.id]);
        }
    }
    p_mutex_.unlock();
}

void TeamMonitor::keyPressEvent(QKeyEvent *event)
{
    switch (event->key())
    {
        case Qt::Key_C:
            p_mutex_.lock();
            players_.clear();
            p_mutex_.unlock();
            break;
        default:
            break;
    }
    this->update();
}