#include <sys/mman.h>
#include <errno.h>
#include <functional>
#include "camera.hpp"
#include "parser/parser.hpp"
#include "configuration.hpp"
#include "class_exception.hpp"
#include <sstream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include "logger.hpp"
#include "core/clock.hpp"

using namespace std;

Camera::Camera(): Sensor("camera")
{
    parser::parse(CONF->get_config_value<string>(CONF->player() + ".camera_info_file"), camera_infos_);
}

bool Camera::start()
{
    if (!this->open())
    {
        return false;
    }

    td_ = std::move(thread(bind(&Camera::run, this)));

    return true;
}

void Camera::set_camera_info(const camera_info &para)
{
    if (!use_mv_)
    {
        return;
    }

    for (auto &item : camera_infos_)
    {
        if (item.second.id == para.id)
        {
            item.second.value = para.value;

            switch (para.id)
            {
                case 1:
                    CameraSetAnalogGain(fd_, para.value);
                    break;

                case 2:
                    CameraSetExposureTime(fd_, para.value * 1000);
                    break;

                default:
                    break;
            }

            break;
        }
    }
}

void Camera::run()
{
    is_alive_ = true;

    if (use_mv_)
    {
        while (is_alive_)
        {
            timestamp_begin = CLOCK->get_timestamp();
            int t1 = sFrameInfo_.uiTimeStamp;
            CameraSdkStatus status = CameraGetImageBuffer(fd_, &sFrameInfo_, &buffer_, 1000);
            timestamp_end = CLOCK->get_timestamp();
            time_used = (sFrameInfo_.uiTimeStamp-t1)*0.1;
            if (status == CAMERA_STATUS_SUCCESS)
            {
                notify(SENSOR_CAMERA);
                CameraReleaseImageBuffer(fd_, buffer_);
            }
            usleep(1000);
        }
    }
    else
    {
        buf_.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf_.memory = V4L2_MEMORY_MMAP;

        while (is_alive_)
        {

            if (ioctl(fd_, VIDIOC_DQBUF, &buf_) == -1)
            {
                LOG(LOG_ERROR) << "VIDIOC_DQBUF failed."<<endll;
                break;
            }

            num_bufs_ = buf_.index;
            buffers_[num_bufs_].bytesused = buf_.bytesused;
            buffers_[num_bufs_].length = buf_.length;
            buffers_[num_bufs_].offset = buf_.m.offset;
            notify(SENSOR_CAMERA);

            if (ioctl(fd_, VIDIOC_QBUF, &buf_) == -1)
            {
                LOG(LOG_ERROR) << "VIDIOC_QBUF error"<<endll;
                break;
            }

            num_bufs_ = buf_.index;

            usleep(10000);
        }
    }
}

void Camera::stop()
{
    is_alive_ = false;
    sleep(1);
    this->close();
    is_open_ = false;
}

void Camera::close()
{
    if (use_mv_)
    {
        if (is_open_)
        {
            CameraUnInit(fd_);
        }
    }
    else
    {
        if (is_open_)
        {
            enum v4l2_buf_type type;
            type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

            if (ioctl(fd_, VIDIOC_STREAMOFF, &type))
            {
                LOG(LOG_ERROR) << "VIDIOC_STREAMOFF error"<<endll;
                return;
            }

            for (num_bufs_ = 0; num_bufs_ < 4; num_bufs_++)
            {
                munmap((void *)(buffers_[num_bufs_].start), buffers_[num_bufs_].length);
                buffers_[num_bufs_].start = nullptr;
            }

            free(buffers_);
            buffers_ = nullptr;
            ::close(fd_);
        }
    }
}

bool Camera::open()
{
    int                     iCameraCounts = 1;
    int                     iStatus = -1;
    tSdkCameraDevInfo       tCameraEnumList;
    use_mv_ = true;
    CameraSdkInit(1);
    iStatus = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);

    if (iCameraCounts == 0)
    {
        use_mv_ = false;
        LOG(LOG_ERROR)<<"open MV camera failed!"<<endll;
    }

    if (use_mv_)
    {
        iStatus = CameraInit(&tCameraEnumList, -1, PARAMETER_TEAM_DEFAULT, &fd_);

        if (iStatus != CAMERA_STATUS_SUCCESS)
        {
            return false;
        }

        CameraGetCapability(fd_, &tCapability_);
        CameraSetAeState(fd_, false);
        CameraSetAnalogGain(fd_, camera_infos_["exposure_gain"].value);
        CameraSetExposureTime(fd_, camera_infos_["exposure_time"].value * 1000);
        CameraSetImageResolution(fd_, &(tCapability_.pImageSizeDesc[0]));
        w_ = tCapability_.pImageSizeDesc[0].iWidth;
        h_ = tCapability_.pImageSizeDesc[0].iHeight;
        CameraPlay(fd_);
    }
    else
    {
        fd_ = ::open(CONF->get_config_value<string>("image.dev_name").c_str(), O_RDWR, 0);

        if (fd_ < 0)
        {
            LOG(LOG_ERROR) << "open camera: " + CONF->get_config_value<string>("image.dev_name") + " failed"<<endll;
            return false;
        }

        w_ = 640;
        h_ = 480;
        v4l2_format fmt;
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.width = w_;
        fmt.fmt.pix.height = h_;
        fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0)
        {
            LOG(LOG_ERROR) << "set format failed"<<endll;
            return false;
        }

        if (ioctl(fd_, VIDIOC_G_FMT, &fmt) < 0)
        {
            LOG(LOG_ERROR) << "get format failed"<<endll;
            return false;
        }

        v4l2_requestbuffers req;
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd_, VIDIOC_REQBUFS, &req) == -1)
        {
            LOG(LOG_ERROR) << "request buffer error"<<endll;
            return false;
        }

        buffers_ = (VideoBuffer *)calloc(req.count, sizeof(VideoBuffer));

        for (num_bufs_ = 0; num_bufs_ < req.count; num_bufs_++)
        {
            buf_.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf_.memory = V4L2_MEMORY_MMAP;
            buf_.index = num_bufs_;

            if (ioctl(fd_, VIDIOC_QUERYBUF, &buf_) == -1)
            {
                LOG(LOG_ERROR) << "query buffer error"<<endll;
                return false;
            }

            buffers_[num_bufs_].length = buf_.length;
            buffers_[num_bufs_].offset = (size_t) buf_.m.offset;
            buffers_[num_bufs_].start = (unsigned char *)mmap(NULL, buf_.length, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fd_, buf_.m.offset);

            if (buffers_[num_bufs_].start == MAP_FAILED)
            {
                int err = errno;
                LOG(LOG_ERROR) << "buffer map error: " << err << endll;
                return false;
            }

            if (ioctl(fd_, VIDIOC_QBUF, &buf_) == -1)
            {
                LOG(LOG_ERROR) << "VIDIOC_QBUF error"<<endll;
                return false;
            }
        }

        enum v4l2_buf_type type;
        type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        if (ioctl(fd_, VIDIOC_STREAMON, &type) == -1)
        {
            LOG(LOG_ERROR) << "VIDIOC_STREAMON error"<<endll;
            return false;
        }
    }

    is_open_ = true;
    return true;
}


Camera::~Camera()
{
    if (td_.joinable())
    {
        td_.join();
    }
}
