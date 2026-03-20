#include "action_monitor.hpp"
#include "configuration.hpp"

using namespace std;
using namespace robot;

ActionMonitor::ActionMonitor()
    : client_(CONF->get_config_value<string>(CONF->player() + ".address"), CONF->get_config_value<int>("net.tcp"),
              bind(&ActionMonitor::data_handler, this, placeholders::_1))
{
    setAttribute(Qt::WA_DeleteOnClose);
    rgl = new RobotGL(ROBOT->get_main_bone(), ROBOT->get_joint_map());
    setCentralWidget(rgl);

    first_connect = true;
    net_info = QString::fromStdString(CONF->get_config_value<string>(CONF->player() + ".address"))
               + ":" + QString::number(CONF->get_config_value<int>("net.tcp"));
    setWindowTitle(net_info);

    timer = new QTimer;
    timer->start(1000);

    connect(timer, &QTimer::timeout, this, &ActionMonitor::procTimer);
    client_.start();
}

void ActionMonitor::data_handler(const tcp_command cmd)
{
    unsigned int size = cmd.size;

    if (cmd.type == JOINT_DATA)
    {
        robot_joint_deg jdeg;
        std::map<int, float> jdegs;

        for (unsigned int i = 0; i < size / sizeof(robot_joint_deg); i++)
        {
            memcpy(&jdeg, cmd.data.c_str() + i * sizeof(robot_joint_deg), sizeof(robot_joint_deg));
            jdegs[jdeg.id] = jdeg.deg;
        }

        rgl->turn_joint(jdegs);
        this->update();
    }
}

void ActionMonitor::procTimer()
{
    if (client_.is_connected())
    {
        if (first_connect)
        {
            client_.regist(JOINT_DATA, DIR_APPLY);
        }

        first_connect = false;
        statusBar()->setStyleSheet("background-color:green");
    }
    else
    {
        first_connect = true;
        statusBar()->setStyleSheet("background-color:red");
    }
}

void ActionMonitor::closeEvent(QCloseEvent *event)
{
    client_.stop();
    emit closed();
}
