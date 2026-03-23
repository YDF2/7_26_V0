#ifndef __IMAGE_MONITOR_HPP
#define __IMAGE_MONITOR_HPP

#include <QtWidgets>
#include <atomic>
#include "robot/robot.hpp"
#include "tcp_client/tcp_client.hpp"
#include "ui/image_label.hpp"
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

class ImageMonitor: public QMainWindow
{
    Q_OBJECT
public:
    ImageMonitor();
    void data_handler(const tcp_command cmd);
public slots:
    void procTimer();
    void procYawSlider(int v);
    void procPitchSlider(int v);
    void procShot(QRect rect);
    void procImageBox(int idx);

signals:
    void closed();

protected:
    void closeEvent(QCloseEvent *event);
    void keyPressEvent(QKeyEvent *event);
private:
    ImageLabel *imageLab;
    QLabel *yawLab, *pitchLab, *netLab;
    QSlider *pitchSlider, *yawSlider;
    QComboBox *sendBox;
    QComboBox *imageBox;
    QCheckBox *sampleBox;
    QTimer *timer;
    tcp_client client_;
    QString net_info;
    bool first_connect;
    cv::Mat curr_image_;
    int width_;
    int height_;
    unsigned int image_count_;
    std::atomic_bool save_;
};

#endif
