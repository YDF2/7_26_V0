#include "image_debuger.hpp"
#include "configuration.hpp"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "imageproc/imageproc.hpp"


using namespace cv;
using namespace std;
using namespace imgproc;

ImageDebuger::ImageDebuger()
{
    this->setAttribute(Qt::WA_DeleteOnClose);
    height_ = CONF->get_config_value<int>("image.height");
    width_ = CONF->get_config_value<int>("image.width");

    srcLab = new ImageLabel(width_, height_);
    dstLab = new ImageLabel(width_, height_);
    curr_image_.create(height_, width_, CV_8UC3);
    curr_index_ = 0;
    infoLab = new QLabel("0/0");
    statusBar()->addWidget(infoLab);

    QHBoxLayout *imageLayout = new QHBoxLayout;
    imageLayout->addWidget(srcLab);
    imageLayout->addWidget(dstLab);

    funcBox = new QComboBox();
    QStringList funclist;
    funclist << "ball & goal";
    funclist << "field";
    funcBox->addItems(funclist);
    btnLoad = new QPushButton("Open Folder");
    btnLast = new QPushButton("Last Frame");
    btnNext = new QPushButton("Next Frame");
    boxAuto = new QCheckBox("Auto Play(ms)");
    boxAuto->setFixedWidth(120);
    delayEdit = new QLineEdit("1000");
    delayEdit->setFixedWidth(50);
    QHBoxLayout *ctrlLayout = new QHBoxLayout;
    ctrlLayout->addWidget(funcBox);
    ctrlLayout->addWidget(btnLoad);
    ctrlLayout->addWidget(btnLast);
    ctrlLayout->addWidget(btnNext);
    ctrlLayout->addWidget(boxAuto);
    ctrlLayout->addWidget(delayEdit);

    frmSlider = new QSlider(Qt::Horizontal);
    frmSlider->setEnabled(false);

    QVBoxLayout *mainLayout = new QVBoxLayout();
    mainLayout->addLayout(imageLayout);
    mainLayout->addLayout(ctrlLayout);
    mainLayout->addWidget(frmSlider);

    QWidget *mainWidget  = new QWidget();
    mainWidget->setLayout(mainLayout);
    this->setCentralWidget(mainWidget);

    timer = new QTimer;
    connect(timer, &QTimer::timeout, this, &ImageDebuger::procTimer);
    connect(btnLoad, &QPushButton::clicked, this, &ImageDebuger::procBtnLoad);
    connect(btnLast, &QPushButton::clicked, this, &ImageDebuger::procBtnLast);
    connect(btnNext, &QPushButton::clicked, this, &ImageDebuger::procBtnNext);
    connect(boxAuto, &QCheckBox::stateChanged, this, &ImageDebuger::procBoxAuto);
    connect(frmSlider, &QSlider::valueChanged, this, &ImageDebuger::procFrmSlider);
    connect(srcLab, &ImageLabel::shot, this, &ImageDebuger::procShot);
}

void ImageDebuger::show_src()
{
    Mat bgr = imread(String((curr_dir_+image_names_.at(curr_index_-1)).toStdString()));
    cvtColor(bgr, rgb_src_, COLOR_BGR2RGB);
    QImage srcImage(rgb_src_.data, rgb_src_.cols, rgb_src_.rows, QImage::Format_RGB888);
    srcLab->set_image(srcImage);
}

void ImageDebuger::show_dst(Mat dst)
{
    //cout<<dst.channels()<<endl;
    QImage dstImage(rgb_src_.data, rgb_src_.cols, rgb_src_.rows, QImage::Format_RGB888);
    dstLab->set_image(dstImage);
    //cout<<"end"<<endl;
}

void ImageDebuger::proc_image(const unsigned int &index)
{
    if (index < 1 || index > image_names_.size())
    {
        return;
    }

    curr_index_ = index;
    infoLab->setText(QString::number(curr_index_) + "/" + QString::number(image_names_.size()));
    frmSlider->setValue(index);
    show_src();

    if (funcBox->currentIndex() == 0) //ball and post detection
    {
        Mat rgb_c = rgb_src_.clone();
        show_dst(rgb_c);
    }
}

void ImageDebuger::procBtnLast()
{
    curr_index_--;

    if (curr_index_ < 1)
    {
        curr_index_ = image_names_.size();
    }

    proc_image(curr_index_);
}

void ImageDebuger::procBtnNext()
{
    curr_index_++;

    if (curr_index_ > image_names_.size())
    {
        curr_index_ = 1;
    }

    proc_image(curr_index_);
}

void ImageDebuger::procBtnLoad()
{
    timer->stop();
    curr_dir_ = QFileDialog::getExistingDirectory(this, "Open image directory", QDir::homePath())+"/";
    if (curr_dir_.isEmpty())
    {
        return;
    }

    QDir dir(curr_dir_);
    QStringList nameFilters;
    nameFilters << "*.jpg" << "*.png";
    image_names_.clear();
    image_names_ = dir.entryList(nameFilters, QDir::Files|QDir::Readable, QDir::Name);
    if (!image_names_.empty())
    {
        frmSlider->setEnabled(true);
        frmSlider->setMinimum(1);
        frmSlider->setMaximum(image_names_.size());
        proc_image(1);
    }
}

void ImageDebuger::procBoxAuto()
{
    if (boxAuto->checkState() == Qt::Checked)
    {
        int delay = delayEdit->text().toInt();

        if (delay < 10)
        {
            delay = 10;
        }

        timer->start(delay);
    }
    else
    {
        timer->stop();
    }
}

void ImageDebuger::procShot(QRect rect)
{
    if (rect.width() > 10 && rect.height() > 10)
    {
        int x, y, w, h;
        x = rect.left();
        y = rect.top();
        w = rect.width();
        h = rect.height();

        if (x + w < width_ && y + h < height_)
        {
            if (funcBox->currentIndex() == 1) // sampling
            {
                Rect mrect(x, y, w, h);
                Mat roi = rgb_src_(mrect);
                imwrite("test.jpg", roi);
                Mat dst;
                rgb_src_.copyTo(dst);

                for (int j = 0; j < height_; j++)
                {
                    for (int i = 0; i < width_; i++)
                    {
                        if (j >= y && j < y + h && i >= x && i < x + w)
                        {
                            continue;
                        }

                        dst.data[j * width_ * 3 + i * 3] = 0;
                        dst.data[j * width_ * 3 + i * 3 + 1] = 0;
                        dst.data[j * width_ * 3 + i * 3 + 2] = 0;
                    }
                }

                show_dst(dst);
            }
        }
    }
}

void ImageDebuger::procFrmSlider(int v)
{
    proc_image(v);
}


void ImageDebuger::procTimer()
{
    procBtnNext();
}

void ImageDebuger::closeEvent(QCloseEvent *event)
{
    emit closed();
}