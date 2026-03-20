#include "image_monitor.hpp"
#include "configuration.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include "common.hpp"

using namespace cv;
using namespace std;
using namespace robot;
using namespace Eigen;

map<string, image_send_type> image_send_types =
    {
        {"origin", IMAGE_SEND_ORIGIN},
        {"result", IMAGE_SEND_RESULT},
        {"point", IMAGE_SEND_POINT},
        {"rect", IMAGE_SEND_POINT},
    };

ImageMonitor::ImageMonitor()
: client_(CONF->get_config_value<string>(CONF->player() + ".address"), CONF->get_config_value<int>("net.tcp"),
              std::bind(&ImageMonitor::data_handler, this, std::placeholders::_1))
{
    this->setAttribute(Qt::WA_DeleteOnClose);
    height_ = CONF->get_config_value<int>("image.height");
    width_ = CONF->get_config_value<int>("image.width");
    first_connect = true;
    save_=false;

    imageLab = new ImageLabel(width_, height_);
    curr_image_.create(height_, width_, CV_8UC3);

    pitchSlider = new QSlider(Qt::Vertical);
    pitchSlider->setRange(-90, 90);
    yawSlider = new QSlider(Qt::Horizontal);
    yawSlider->setRange(-90, 90);

    sampleBox = new QCheckBox("Image Sample");
    imageBox = new QComboBox();
    QStringList func;
    func<<"origin"<<"result";
    imageBox->addItems(func);
    imageBox->setCurrentIndex(1);

    sendBox = new QComboBox();
    func.clear();
    func<<"none"<<"point"<<"rect";
    sendBox->addItems(func);
    sendBox->setCurrentIndex(0);

    QHBoxLayout *imageLayout = new QHBoxLayout();
    imageLayout->addWidget(new QLabel("image"));
    imageLayout->addWidget(imageBox);

    QHBoxLayout *sendLayout = new QHBoxLayout();
    sendLayout->addWidget(new QLabel("send"));
    sendLayout->addWidget(sendBox);

    QVBoxLayout *leftLayout = new QVBoxLayout();
    leftLayout->addWidget(imageLab);
    leftLayout->addWidget(yawSlider);

    QVBoxLayout *ctrlLayout = new QVBoxLayout;
    ctrlLayout->addWidget(sampleBox);
    ctrlLayout->addLayout(imageLayout);
    ctrlLayout->addLayout(sendLayout);

    QHBoxLayout *mainLayout = new QHBoxLayout();
    mainLayout->addLayout(leftLayout);
    mainLayout->addWidget(pitchSlider);
    mainLayout->addLayout(ctrlLayout);

    QWidget *mainWidget  = new QWidget();
    mainWidget->setLayout(mainLayout);
    this->setCentralWidget(mainWidget);

    yawLab = new QLabel();
    pitchLab = new QLabel();
    netLab = new QLabel();
    netLab->setFixedWidth(100);

    statusBar()->addWidget(pitchLab);
    statusBar()->addWidget(yawLab);
    statusBar()->addWidget(netLab);

    net_info = QString::fromStdString(CONF->get_config_value<string>(CONF->player() + ".address"))
               + ":" + QString::number(CONF->get_config_value<int>("net.tcp"));
    setWindowTitle(net_info);

    timer = new QTimer;
    timer->start(1000);
    image_count_=0;

    connect(timer, &QTimer::timeout, this, &ImageMonitor::procTimer);
    connect(yawSlider, &QSlider::valueChanged, this, &ImageMonitor::procYawSlider);
    connect(pitchSlider, &QSlider::valueChanged, this, &ImageMonitor::procPitchSlider);
    connect(imageLab, &ImageLabel::shot, this, &ImageMonitor::procShot);
    connect(imageBox, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &ImageMonitor::procImageBox);
    client_.start();
    yawSlider->setEnabled(false);
    pitchSlider->setEnabled(false);
}

void ImageMonitor::data_handler(const tcp_command cmd)
{
    if (cmd.type == IMG_DATA)
    {
        vector<unsigned char> buf(cmd.size);
        memcpy(&buf[0], cmd.data.c_str(), cmd.size);
        try
        {
            Mat bgr = imdecode(buf, cv::IMREAD_COLOR);
            if(save_||(sampleBox->isChecked()&&image_count_++%10==0))
            {
                imwrite(String(get_time()+".png"), bgr);
                save_=false;
            }
            Mat dst;
            cvtColor(bgr, dst, COLOR_BGR2RGB);
            QImage dstImage((const unsigned char *)(dst.data), dst.cols, dst.rows, QImage::Format_RGB888);
            imageLab->set_image(dstImage);
            bgr.release();
            dst.release();
        }
        catch (std::exception &e)
        {
            cout << e.what() << endl;
        }
    }
    else if(cmd.type == REMOTE_DATA)
    {
        remote_data_type t1;
        memcpy(&t1, cmd.data.c_str(), enum_size);
        if(t1==IMAGE_SEND_TYPE)
        {
            image_send_type t2;
            memcpy(&t2, cmd.data.c_str()+enum_size, enum_size);
            if(t2==IMAGE_SEND_DIS)
            {
                float x,y;
                memcpy(&x, cmd.data.c_str()+2*enum_size, float_size);
                memcpy(&y, cmd.data.c_str()+2*enum_size+float_size, float_size);

            }
        }
    }
}

void ImageMonitor::procTimer()
{
    if (client_.is_connected())
    {
        if (first_connect)
        {
            client_.regist(IMG_DATA, DIR_APPLY);
            usleep(10000);
            client_.regist(REMOTE_DATA, DIR_BOTH);
        }

        first_connect = false;
        netLab->setStyleSheet("background-color:green");
        yawSlider->setEnabled(true);
        pitchSlider->setEnabled(true);
        imageBox->setEnabled(true);
        sendBox->setEnabled(true);
    }
    else
    {
        first_connect = true;
        netLab->setStyleSheet("background-color:red");
        yawSlider->setEnabled(false);
        pitchSlider->setEnabled(false);
        imageBox->setEnabled(false);
        sendBox->setEnabled(false);
    }
}

void ImageMonitor::procShot(QRect rect)
{
    if (rect.width() > 5 && rect.height() > 5)
    {
        int x, y, w, h;
        x = rect.left();
        y = rect.top();
        w = rect.width();
        h = rect.height();
        QString str=sendBox->currentText();
        if (str=="rect")
        {
            remote_data_type t = IMAGE_SEND_TYPE;
            image_send_type st = IMAGE_SEND_RECT;
            tcp_command cmd;
            cmd.type = REMOTE_DATA;
            cmd.size = 4 * int_size + 2*enum_size;
            cmd.data.clear();
            cmd.data.append((char *)(&t), enum_size);
            cmd.data.append((char *)(&st), enum_size);
            cmd.data.append((char *)(&x), int_size);
            cmd.data.append((char *)(&y), int_size);
            cmd.data.append((char *)(&w), int_size);
            cmd.data.append((char *)(&h), int_size);
            client_.write(cmd);
        }
        else if (str=="point")
        {
            x=x+w/2;
            y=y+h/2;
            remote_data_type t = IMAGE_SEND_TYPE;
            image_send_type st = IMAGE_SEND_POINT;
            tcp_command cmd;
            cmd.type = REMOTE_DATA;
            cmd.size = 2 * int_size + 2*enum_size;
            cmd.data.clear();
            cmd.data.append((char *)(&t), enum_size);
            cmd.data.append((char *)(&st), enum_size);
            cmd.data.append((char *)(&x), int_size);
            cmd.data.append((char *)(&y), int_size);
            client_.write(cmd);
        }
    }
}

void ImageMonitor::procImageBox(int idx)
{
    remote_data_type t = IMAGE_SEND_TYPE;
    tcp_command cmd;
    image_send_type st = image_send_types[imageBox->itemText(idx).toStdString()];
    cmd.type = REMOTE_DATA;
    cmd.size = 2*enum_size;
    cmd.data.clear();
    cmd.data.append((char *)(&t), enum_size);
    cmd.data.append((char *)(&st), enum_size);
    client_.write(cmd);
}

void ImageMonitor::procPitchSlider(int v)
{
    pitchLab->setText(QString::number(v));
    float yaw = -(float)(yawSlider->value());
    float pitch = -(float)v;
    remote_data_type t = LOOKAT_DATA;

    tcp_command cmd;
    cmd.type = REMOTE_DATA;
    cmd.size = 2 * float_size + enum_size;
    cmd.data.clear();
    cmd.data.append((char *)(&t), enum_size);
    cmd.data.append((char *)(&yaw), float_size);
    cmd.data.append((char *)(&pitch), float_size);
    client_.write(cmd);
}

void ImageMonitor::procYawSlider(int v)
{
    yawLab->setText(QString::number(v));
    float yaw = -(float)v;
    float pitch = -(float)(pitchSlider->value());
    remote_data_type t = LOOKAT_DATA;

    tcp_command cmd;
    cmd.type = REMOTE_DATA;
    cmd.size = 2 * float_size + enum_size;
    cmd.data.clear();
    cmd.data.append((char *)(&t), enum_size);
    cmd.data.append((char *)(&yaw), float_size);
    cmd.data.append((char *)(&pitch), float_size);
    client_.write(cmd);
}

void ImageMonitor::closeEvent(QCloseEvent *event)
{
    client_.stop();
    emit closed();
}

void ImageMonitor::keyPressEvent(QKeyEvent *event)
{
    switch (event->key())
    {
        case Qt::Key_C:
            save_=true;
            break;
        default:
            break;
    }
    this->update();
}