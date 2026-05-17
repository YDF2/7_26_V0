#ifndef __FIELD_MONITOR_HPP
#define __FIELD_MONITOR_HPP

#include <QtWidgets>
#include "tcp_client/tcp_client.hpp"
#include <opencv2/opencv.hpp>

class FieldMonitor: public QMainWindow
{
    Q_OBJECT
public:
    FieldMonitor();
    void data_handler(const tcp_command cmd);

signals:
    void closed();
    void localizationUpdated();

protected:
    void paintEvent(QPaintEvent *event);
    void closeEvent(QCloseEvent *event);

private:
    void drawField(QPainter &p);
    void drawRobot(QPainter &p);
    void drawBall(QPainter &p);

    tcp_client client_;

    // 场地参数 (cm) — 标准 RoboCup Kidsize 决赛 600×400
    int field_length_;
    int field_width_;
    int goal_width_;
    int goal_depth_;
    int penalty_area_length_;    // 大禁区纵深
    int penalty_area_width_;     // 大禁区宽
    int goal_area_length_;       // 小禁区纵深
    int goal_area_width_;        // 小禁区宽
    int penalty_mark_distance_;
    int center_circle_diameter_;
    int border_strip_width_;

    // 机器人
    float robot_x_, robot_y_, robot_dir_;
    bool  robot_valid_;

    // 球
    float ball_x_, ball_y_;
    bool  ball_can_see_;
    bool  ball_valid_;

    // 显示
    float scale_;
    int   robot_sq_;   // 方块半边长 (px)
    int   ball_r_;     // 球半径 (px)

    QLabel *robotLabel, *ballLabel;
    QLabel *netLab;
    bool    first_connect_;
    QTimer *timer;
};

#endif
