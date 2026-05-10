#ifndef __WALK_REMOTE_HPP
#define __WALK_REMOTE_HPP

#include <QtWidgets>
#include "tcp_client/tcp_client.hpp"

class WalkRemote : public QMainWindow
{
    Q_OBJECT
public:
    WalkRemote(QWidget *parent = nullptr);
    void updateLab();
public slots:
    void procTimer();
    void procXSlider(int v);
    void procYSlider(int v);
    void procDSlider(int v);
    
signals:
    void closed();

protected:
    void closeEvent(QCloseEvent *event);

private:
    QCheckBox *startCheck;
    QRadioButton *btnSpot, *btnRand, *btnMan;
    QSlider *dirSlider, *xSlider, *ySlider;
    QLabel *dirLab, *xLab, *yLab;
    float _x, _y, _dir;
    bool _enable;
    int range_;
    float scale_d;
    float scale_xy;
    QTimer *timer;
    QString netinfo_;
    tcp_client client_;
    bool first_connect;
};

#endif
