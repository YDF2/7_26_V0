#include "debuger.hpp"
#include "configuration.hpp"
#include "robot/robot.hpp"
#include "image_monitor/image_monitor.hpp"
#include "image_debuger/image_debuger.hpp"
#include "camera_setter/camera_setter.hpp"
#include "team_monitor/team_monitor.hpp"
#include "action_debuger/action_debuger.hpp"
#include "action_monitor/action_monitor.hpp"
#include "walk_remote/walk_remote.hpp"
#include "joint_revise/joint_revise.hpp"
#include "init_pos_editor/init_pos_editor.hpp"
#include <algorithm>
#include <cmath>

using namespace std;
using namespace robot;

Debuger::Debuger()
{
    QVBoxLayout *mainLayout = new QVBoxLayout;
    
    QHBoxLayout *pLayout = new QHBoxLayout;
    pBox = new QComboBox();
    vector<int> players = CONF->players();
    for(auto id:players)
        pBox->addItem(QString::number(id));
    if(!players.empty())
        current_player = players[0];
    pLayout->addWidget(new QLabel("Player:"));
    pLayout->addWidget(pBox);
    mainLayout->addLayout(pLayout);

    QVBoxLayout *leftLayout, *rightLayout;
    leftLayout = new QVBoxLayout;
    rightLayout = new QVBoxLayout;
    QPushButton *btnDebuger;
    btnDebuger = new QPushButton("Image Monitor");
    connect(btnDebuger, &QPushButton::clicked, this, &Debuger::procBtnIM);
    leftLayout->addWidget(btnDebuger);
    btnDebuger = new QPushButton("Image Debuger");
    connect(btnDebuger, &QPushButton::clicked, this, &Debuger::procBtnID);
    leftLayout->addWidget(btnDebuger);
    btnDebuger = new QPushButton("Camera Setter");
    connect(btnDebuger, &QPushButton::clicked, this, &Debuger::procBtnCS);
    leftLayout->addWidget(btnDebuger);
    btnDebuger = new QPushButton("Team Monitor");
    connect(btnDebuger, &QPushButton::clicked, this, &Debuger::procBtnTM);
    leftLayout->addWidget(btnDebuger);
    btnDebuger = new QPushButton("Init Pos Editor");

    connect(btnDebuger, &QPushButton::clicked, this, &Debuger::procBtnIPE);
    rightLayout->addWidget(btnDebuger);
    btnDebuger = new QPushButton("Action Monitor");
    connect(btnDebuger, &QPushButton::clicked, this, &Debuger::procBtnAM);
    rightLayout->addWidget(btnDebuger);
    btnDebuger = new QPushButton("Action Debuger");
    connect(btnDebuger, &QPushButton::clicked, this, &Debuger::procBtnAD);
    rightLayout->addWidget(btnDebuger);
    btnDebuger = new QPushButton("Walk Remote");
    connect(btnDebuger, &QPushButton::clicked, this, &Debuger::procBtnWR);
    rightLayout->addWidget(btnDebuger);
    btnDebuger = new QPushButton("Joint Revise");
    connect(btnDebuger, &QPushButton::clicked, this, &Debuger::procBtnJR);
    rightLayout->addWidget(btnDebuger);

    QHBoxLayout *btnLayout = new QHBoxLayout;
    btnLayout->addLayout(leftLayout);
    btnLayout->addLayout(rightLayout);
    mainLayout->addLayout(btnLayout);

    QWidget *mainWidget = new QWidget;
    mainWidget->setLayout(mainLayout);
    this->setCentralWidget(mainWidget);

    connect(pBox, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &Debuger::procPlayerChanged);

    opened_debuger_count = 0;

    QSize hint_size = this->sizeHint();
    if(hint_size.isValid())
        resize(hint_size.width() * 2, hint_size.height() * 2);
    base_window_size = size();
    cacheBaseFonts();

    procPlayerChanged(0);
}

void Debuger::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);

    if(!base_window_size.isValid() || base_fonts.empty())
        return;

    double scale_x = static_cast<double>(event->size().width()) / base_window_size.width();
    double scale_y = static_cast<double>(event->size().height()) / base_window_size.height();
    double scale = std::min(scale_x, scale_y);
    applyScale(scale);
}

void Debuger::cacheBaseFonts()
{
    base_fonts.clear();
    base_fonts[this] = this->font();

    const auto widgets = findChildren<QWidget*>();
    for(QWidget *widget : widgets)
        base_fonts[widget] = widget->font();
}

void Debuger::applyScale(double scale)
{
    scale = std::max(0.5, std::min(scale, 3.0));

    for(auto it = base_fonts.begin(); it != base_fonts.end(); ++it)
    {
        QWidget *widget = it.key();
        if(!widget)
            continue;

        QFont font = it.value();
        if(font.pointSizeF() > 0)
            font.setPointSizeF(std::max(1.0, font.pointSizeF() * scale));
        else if(font.pixelSize() > 0)
            font.setPixelSize(std::max(1, static_cast<int>(std::round(font.pixelSize() * scale))));

        widget->setFont(font);
    }
}

void Debuger::procPlayerChanged(int index)
{
    current_player = atoi(pBox->currentText().toStdString().c_str());
    CONF->init(current_player);
    ROBOT->init(CONF->robot_file(), CONF->action_file(), CONF->offset_file());
}

void Debuger::procClosed()
{
    opened_debuger_count--;
    if(opened_debuger_count == 0)
     pBox->setEnabled(true);
}

void Debuger::procBtnIM()
{
    ImageMonitor *foo = new ImageMonitor();
    connect(foo, &ImageMonitor::closed, this, &Debuger::procClosed);
    foo->show();
    opened_debuger_count++;
    pBox->setEnabled(false);
}

void Debuger::procBtnID()
{
    ImageDebuger *foo = new ImageDebuger();
    connect(foo, &ImageDebuger::closed, this, &Debuger::procClosed);
    foo->show();
    opened_debuger_count++;
    pBox->setEnabled(false);
}

void Debuger::procBtnCS()
{
    CameraSetter *foo = new CameraSetter();
    connect(foo, &CameraSetter::closed, this, &Debuger::procClosed);
    foo->show();
    opened_debuger_count++;
    pBox->setEnabled(false);
}

void Debuger::procBtnTM()
{
    TeamMonitor *foo = new TeamMonitor();
    connect(foo, &TeamMonitor::closed, this, &Debuger::procClosed);
    foo->show();
    opened_debuger_count++;
    pBox->setEnabled(false);
}

void Debuger::procBtnAD()
{
    ActionDebuger *foo = new ActionDebuger();
    connect(foo, &ActionDebuger::closed, this, &Debuger::procClosed);
    foo->show();
    opened_debuger_count++;
    pBox->setEnabled(false);
}

void Debuger::procBtnAM()
{
    ActionMonitor *foo = new ActionMonitor();
    connect(foo, &ActionMonitor::closed, this, &Debuger::procClosed);
    foo->show();
    opened_debuger_count++;
    pBox->setEnabled(false);
}

void Debuger::procBtnWR()
{
    WalkRemote *foo = new WalkRemote();
    connect(foo, &WalkRemote::closed, this, &Debuger::procClosed);
    foo->show();
    opened_debuger_count++;
    pBox->setEnabled(false);
}

void Debuger::procBtnIPE()
{
    InitPosEditor *foo = new InitPosEditor(current_player);
    connect(foo, &InitPosEditor::closed, this, &Debuger::procClosed);
    foo->show();
    opened_debuger_count++;
    pBox->setEnabled(false);
}

void Debuger::procBtnJR()
{
    JointRevise *foo = new JointRevise();
    connect(foo, &JointRevise::closed, this, &Debuger::procClosed);
    foo->show();
    opened_debuger_count++;
    pBox->setEnabled(false);
}
