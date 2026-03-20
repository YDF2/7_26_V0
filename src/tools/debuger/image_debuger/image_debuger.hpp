#ifndef __IMAGE_DEBUGER_HPP
#define __IMAGE_DEBUGER_HPP

#include <QtWidgets>
#include "robot/robot.hpp"
#include "ui/image_label.hpp"
#include <opencv2/opencv.hpp>
#include "model.hpp"

class ImageDebuger: public QMainWindow
{
    Q_OBJECT
public:
    ImageDebuger();
public slots:
    void procTimer();
    void procBtnLoad();
    void procBtnLast();
    void procBtnNext();
    void procBoxAuto();
    void procShot(QRect rect);
    void procFrmSlider(int v);

signals:
    void closed();

protected:
    void closeEvent(QCloseEvent *event);
    
private:
    void proc_image(const unsigned int &index);
    void show_src();
    void show_dst(cv::Mat dst);

    QPushButton *btnLoad, *btnNext, *btnLast;
    QCheckBox *boxAuto;
    ImageLabel *srcLab, *dstLab;
    QLabel *infoLab;
    QTimer *timer;
    QSlider *frmSlider;
    QLineEdit *delayEdit;
    QComboBox *funcBox;
    unsigned int curr_index_;
    cv::Mat curr_image_;
    QString curr_dir_;
    QStringList image_names_;
    cv::Mat rgb_src_;
    int width_;
    int height_;
};

#endif
