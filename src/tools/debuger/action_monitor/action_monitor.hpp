#ifndef __ACTION_MONITOR_HPP
#define __ACTION_MONITOR_HPP

#include <QtWidgets>
#include "ui/robot_gl.hpp"
#include "robot/robot.hpp"
#include "tcp_client/tcp_client.hpp"

class ActionMonitor: public QMainWindow
{
    Q_OBJECT
public:
    ActionMonitor();
    void data_handler(const tcp_command cmd);
public slots:
    void procTimer();
signals:
    void closed();
    
protected:
    void closeEvent(QCloseEvent *event);
private:
    RobotGL *rgl;
    QTimer *timer;
    tcp_client client_;
    QString net_info;
    bool first_connect;
};

#endif
