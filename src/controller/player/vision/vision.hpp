#ifndef __VISION_HPP
#define __VISION_HPP

#include <queue>
#include <atomic>
#include <mutex>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "timer.hpp"
#include "observer.hpp"
#include "sensor/camera.hpp"
#include "configuration.hpp"
#include "options/options.hpp"
#include "common.hpp"
#include "singleton.hpp"
#include "tcp.hpp"
#include "common.hpp"
#include "math/math.hpp"
#include "sensor/imu.hpp"
#include "sensor/motor.hpp"
#include "localization/localization.h"
#include "trt_detector.hpp"

class Vision: public Timer, public Subscriber, public Singleton<Vision>
{
public:
    Vision();
    ~Vision();
    int w(){return w_;}
    int h(){return h_;}
    void updata(const pub_ptr &pub, const int &type);
    bool start();
    void stop();
    void set_camera_info(const camera_info &para);
    void get_point_dis(int x, int y);

    void set_img_send_type(image_send_type t)
    {
        img_sd_type_ = t;
    }

public:
    std::atomic_bool localization_;
    std::atomic_bool can_see_post_;
private:
    Eigen::Vector2d odometry(const Eigen::Vector2i &pos, const robot_math::TransformMatrix &camera_matrix);
    Eigen::Vector2d camera2self(const Eigen::Vector2d &pos, double head_yaw);
    Imu::imu_data imu_data_vison;//修改
    float dir=0;//修改
private:
    void run();
    void send_image(const cv::Mat &src);
    
    std::queue<Imu::imu_data> imu_datas_;
    std::queue< std::vector<double> > foot_degs_, head_degs_;
    std::queue<int> spfs_;
    
    float head_yaw_, head_pitch_;

    Eigen::Vector2d odometry_offset_;

    bool detect_filed_;
    bool use_mv_;
    int p_count_;
    int w_, h_;
    int camera_w_,  camera_h_, camera_size_;
    std::map<std::string, camera_info> camera_infos_;
    camera_param params_;
    robot_math::TransformMatrix camera_matrix_;
    
    std::vector<object_det> ball_dets_, post_dets_; 
    float d_w_h_;
    int ball_id_, post_id_;
    float ball_prob_, post_prob_;
    int min_ball_w_, min_ball_h_;
    int min_post_w_, min_post_h_;

    int cant_see_ball_count_;
    int can_see_post_count_;

    bool is_busy_;
    bool zed_cam_param_applied_;
    image_send_type img_sd_type_;

    TRTDetector detector_;

    unsigned char *dev_src_;
    unsigned char *dev_bgr_;
    unsigned char *dev_ori_;
    unsigned char *dev_sized_;
    unsigned char *dev_undis_;
    unsigned char *dev_yuyv_;
    unsigned char *camera_src_;
    float *dev_rgbfp_;
    int src_size_;
    int bgr_size_;
    int ori_size_;
    int yuyv_size_;
    int sized_size_;
    int rgbf_size_;

    cv::Mat camK;
	cv::Mat newCamK;
	cv::Mat invCamK;
	cv::Mat D;
	cv::Mat R;

    cv::Mat mapx;
	cv::Mat mapy;

    float *pCamKData;
	float *pInvNewCamKData;
	float *pDistortData;
	float *pMapxData;
	float *pMapyData;
    
    mutable std::mutex frame_mtx_, imu_mtx_;
};

#define VISION Vision::instance()

#endif
