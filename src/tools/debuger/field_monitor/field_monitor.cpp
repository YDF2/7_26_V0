#include "field_monitor.hpp"
#include "configuration.hpp"

using namespace std;

// 场地坐标系：
//   field x → 长度方向 (600cm, 球门在左右两端) → 窗口 x
//   field y → 宽度方向 (400cm)                  → 窗口 y (翻转)

FieldMonitor::FieldMonitor()
    : client_(CONF->get_config_value<string>(CONF->player() + ".address"),
              CONF->get_config_value<int>("net.tcp"),
              std::bind(&FieldMonitor::data_handler, this, std::placeholders::_1))
{
    setAttribute(Qt::WA_DeleteOnClose);

    field_length_           = 600;
    field_width_            = 400;
    goal_width_             = 150;
    goal_depth_             = 50;
    penalty_area_length_    = 100;
    penalty_area_width_     = 250;
    goal_area_length_       = 50;
    goal_area_width_        = 150;
    penalty_mark_distance_  = 100;
    center_circle_diameter_ = 120;
    border_strip_width_     = 70;

    robot_x_ = robot_y_ = robot_dir_ = 0;
    robot_valid_ = false;
    ball_x_ = ball_y_ = 0;
    ball_can_see_ = false;
    ball_valid_ = false;

    scale_       = 1.0f;
    robot_sq_    = 6;
    ball_r_      = 5;

    first_connect_ = true;

    int base_w = field_length_ + 2 * border_strip_width_;   // 740
    int base_h = field_width_  + 2 * border_strip_width_;   // 540
    resize(static_cast<int>(base_w * 1.2), static_cast<int>(base_h * 1.2));
    setMinimumSize(500, 400);

    setStyleSheet("background:rgb(0,110,0)");

    robotLabel = new QLabel("robot: --");
    ballLabel  = new QLabel("ball: --");
    netLab = new QLabel();
    netLab->setFixedWidth(100);
    statusBar()->addWidget(robotLabel);
    statusBar()->addWidget(ballLabel);
    statusBar()->addWidget(netLab);

    QString net_info = QString::fromStdString(
        CONF->get_config_value<string>(CONF->player() + ".address"))
        + ":" + QString::number(CONF->get_config_value<int>("net.tcp"));
    setWindowTitle("Field Monitor - " + net_info);

    connect(this, &FieldMonitor::localizationUpdated, this,
            static_cast<void (FieldMonitor::*)()>(&FieldMonitor::update));

    timer = new QTimer;
    timer->start(1000);
    connect(timer, &QTimer::timeout, this, [this]() {
        if (client_.is_connected())
        {
            if (first_connect_)
            {
                client_.regist(REMOTE_DATA, DIR_APPLY);
                usleep(10000);
            }
            first_connect_ = false;
            netLab->setStyleSheet("background-color:green");
        }
        else
        {
            first_connect_ = true;
            netLab->setStyleSheet("background-color:red");
        }
    });

    client_.start();
}

void FieldMonitor::data_handler(const tcp_command cmd)
{
    if (cmd.type == REMOTE_DATA)
    {
        if (cmd.size < enum_size)
            return;
        remote_data_type t1;
        memcpy(&t1, cmd.data.c_str(), enum_size);
        if (t1 == LOCALIZATION_DATA)
        {
            std::cout << "FM: recv LOCALIZATION_DATA size=" << cmd.size << std::endl;
            if (cmd.size < enum_size + 7 * float_size + bool_size)
                return;
            float sx, sy, sdir;
            float bx, by, bself_x, bself_y;
            bool can_see;
            memcpy(&sx,    cmd.data.c_str() + enum_size,                       float_size);
            memcpy(&sy,    cmd.data.c_str() + enum_size + float_size,           float_size);
            memcpy(&sdir,  cmd.data.c_str() + enum_size + 2 * float_size,       float_size);
            memcpy(&bx,    cmd.data.c_str() + enum_size + 3 * float_size,       float_size);
            memcpy(&by,    cmd.data.c_str() + enum_size + 4 * float_size,       float_size);
            memcpy(&bself_x, cmd.data.c_str() + enum_size + 5 * float_size,     float_size);
            memcpy(&bself_y, cmd.data.c_str() + enum_size + 6 * float_size,     float_size);
            memcpy(&can_see, cmd.data.c_str() + enum_size + 7 * float_size,     bool_size);

            robot_x_ = sx;
            robot_y_ = sy;
            robot_dir_ = sdir;
            robot_valid_ = true;

            ball_x_ = bx;
            ball_y_ = by;
            ball_can_see_ = can_see;
            ball_valid_ = true;

            emit localizationUpdated();
        }
    }
}

void FieldMonitor::paintEvent(QPaintEvent *)
{
    int bw = field_length_ + 2 * border_strip_width_;
    int bh = field_width_  + 2 * border_strip_width_;
    float sx = static_cast<float>(width())  / bw;
    float sy = static_cast<float>(height()) / bh;
    scale_ = std::min(sx, sy);

    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.translate(width() / 2.0, height() / 2.0);
    p.scale(scale_, scale_);

    drawField(p);
    drawBall(p);
    drawRobot(p);

    if (robot_valid_)
        robotLabel->setText(QString("robot: x=%1 y=%2 dir=%3°")
            .arg(robot_x_, 6, 'f', 1)
            .arg(robot_y_, 6, 'f', 1)
            .arg(robot_dir_, 5, 'f', 1));
    else
        robotLabel->setText("robot: --");

    if (ball_valid_)
    {
        QString vis = ball_can_see_ ? "O" : "X";
        ballLabel->setText(QString("ball: x=%1 y=%2 %3")
            .arg(ball_x_, 6, 'f', 1)
            .arg(ball_y_, 6, 'f', 1)
            .arg(vis));
    }
    else
        ballLabel->setText("ball: --");
}

// ─── 坐标映射：field(x,y) → window(x,-y) ───
static inline int fx(int field_x) { return field_x; }
static inline int fy(int field_y) { return -field_y; }

void FieldMonitor::drawField(QPainter &p)
{
    int fl2 = field_length_ / 2;    // 300
    int fw2 = field_width_  / 2;    // 200

    // ── 边线 ──
    p.setPen(QPen(Qt::white, 2));
    p.setBrush(Qt::NoBrush);
    p.drawRect(fx(-fl2), fy( fw2), field_length_, field_width_);

    // ── 中线 ──
    p.drawLine(fx(0), fy(-fw2), fx(0), fy(fw2));

    // ── 中圈 ──
    int cr = center_circle_diameter_ / 2;
    p.drawEllipse(fx(-cr), fy( cr), center_circle_diameter_, center_circle_diameter_);

    // ── 开球点 ──
    p.setBrush(Qt::white);
    p.drawEllipse(fx(-2), fy(2), 4, 4);

    // ── 大禁区(左, x=-fl2 端) ──
    int pal2 = penalty_area_width_ / 2;
    int pal  = penalty_area_length_;
    p.setBrush(Qt::NoBrush);
    p.drawLine(fx(-fl2), fy(-pal2), fx(-fl2 + pal), fy(-pal2));
    p.drawLine(fx(-fl2), fy( pal2), fx(-fl2 + pal), fy( pal2));
    p.drawLine(fx(-fl2 + pal), fy(-pal2), fx(-fl2 + pal), fy(pal2));

    // ── 小禁区(左) ──
    int gal2 = goal_area_width_ / 2;
    int gal  = goal_area_length_;
    p.drawLine(fx(-fl2), fy(-gal2), fx(-fl2 + gal), fy(-gal2));
    p.drawLine(fx(-fl2), fy( gal2), fx(-fl2 + gal), fy( gal2));
    p.drawLine(fx(-fl2 + gal), fy(-gal2), fx(-fl2 + gal), fy(gal2));

    // ── 大禁区(右, x=+fl2 端) ──
    p.drawLine(fx(fl2), fy(-pal2), fx(fl2 - pal), fy(-pal2));
    p.drawLine(fx(fl2), fy( pal2), fx(fl2 - pal), fy( pal2));
    p.drawLine(fx(fl2 - pal), fy(-pal2), fx(fl2 - pal), fy(pal2));

    // ── 小禁区(右) ──
    p.drawLine(fx(fl2), fy(-gal2), fx(fl2 - gal), fy(-gal2));
    p.drawLine(fx(fl2), fy( gal2), fx(fl2 - gal), fy( gal2));
    p.drawLine(fx(fl2 - gal), fy(-gal2), fx(fl2 - gal), fy(gal2));

    // ── 球门(左) ──
    int gw2 = goal_width_ / 2;
    p.drawRect(fx(-fl2 - goal_depth_), fy(gw2), goal_depth_, goal_width_);

    // ── 球门(右) ──
    p.drawRect(fx(fl2), fy(gw2), goal_depth_, goal_width_);

    // ── 罚球点(左) ──
    int pm = penalty_mark_distance_;
    p.setBrush(Qt::white);
    p.drawEllipse(fx(-fl2 + pm - 2), fy(-2), 4, 4);

    // ── 罚球点(右) ──
    p.drawEllipse(fx(fl2 - pm - 2), fy(-2), 4, 4);
}

void FieldMonitor::drawBall(QPainter &p)
{
    if (!ball_valid_)
        return;

    int bx = fx(static_cast<int>(ball_x_));
    int by = fy(static_cast<int>(ball_y_));

    int r = ball_r_;
    if (ball_can_see_)
    {
        p.setPen(QPen(QColor(255, 140, 0), 2));
        p.setBrush(QBrush(QColor(255, 140, 0)));
    }
    else
    {
        p.setPen(QPen(Qt::gray, 1, Qt::DotLine));
        p.setBrush(Qt::NoBrush);
    }
    p.drawEllipse(bx - r, by - r, r * 2, r * 2);
}

void FieldMonitor::drawRobot(QPainter &p)
{
    if (!robot_valid_)
        return;

    int rx = fx(static_cast<int>(robot_x_));
    int ry = fy(static_cast<int>(robot_y_));

    p.save();
    p.translate(rx, ry);
    p.rotate(-static_cast<double>(robot_dir_));

    int sq = robot_sq_;
    p.setPen(QPen(Qt::white, 2));
    p.setBrush(QBrush(Qt::white));
    p.drawRect(-sq, -sq, sq * 2, sq * 2);

    p.drawLine(0, 0, sq + 8, 0);

    p.restore();
}

void FieldMonitor::closeEvent(QCloseEvent *event)
{
    client_.stop();
    emit closed();
}
