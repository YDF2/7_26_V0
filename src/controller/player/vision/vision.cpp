#include "vision.hpp"
#include "parser/parser.hpp"
#include <cuda_runtime.h>
#include "server/server.hpp"
#include <algorithm>
#include "core/worldmodel.hpp"
#include "imageproc/imageproc.hpp"
#include "cuda_utils.hpp"
#include <fstream>

using namespace std;
using namespace cv;
using namespace robot_math;
using namespace Eigen;
using namespace robot;
using namespace imgproc;

static inline void check_error(cudaError_t status)
{
    check_error_cuda(status);
}

Vision::Vision() : Timer(CONF->get_config_value<int>("vision_period"))
{
    p_count_ = 0;
    cant_see_ball_count_ = 0;
    can_see_post_count_ = 0;
    is_busy_ = false;
    zed_cam_param_applied_ = false;
    use_zed_ = false;
    w_ = CONF->get_config_value<int>("image.width");
    h_ = CONF->get_config_value<int>("image.height");
    img_sd_type_ = IMAGE_SEND_RESULT;
    camera_src_ = nullptr;

    parser::parse(CONF->get_config_value<string>(CONF->player() + ".camera_info_file"), camera_infos_);
    parser::parse(CONF->get_config_value<string>(CONF->player() + ".camera_params_file"), params_);
    LOG(LOG_INFO) << setw(12) << "algorithm:" << setw(18) << "[vision]"
                  << " started!" << endll;
    ball_id_ = 1;
    post_id_ = 0;
    ball_prob_ = CONF->get_config_value<float>("detection.ball");
    post_prob_ = CONF->get_config_value<float>("detection.post");
    min_ball_w_ = CONF->get_config_value<int>("detection.ball_w");
    min_ball_h_ = CONF->get_config_value<int>("detection.ball_h");
    min_post_w_ = CONF->get_config_value<int>("detection.post_w");
    min_post_h_ = CONF->get_config_value<int>("detection.post_h");
    odometry_offset_ = CONF->get_config_vector<double, 2>(CONF->player() + ".odometry_offset");
    odometry_offset_.x() = deg2rad(odometry_offset_.x());
    odometry_offset_.y() = deg2rad(odometry_offset_.y());

    camK = cv::Mat::eye(3, 3, CV_32F);
    newCamK = cv::Mat::eye(3, 3, CV_32F);
    invCamK = cv::Mat::eye(3, 3, CV_32F);
    float rotation_mat[9] = {0.999956, 0.00837928, 0.00423897,
                             -0.00839916, 0.999954, 0.00469485,
                             -0.00419943, -0.00473025, 0.99998};
    R = cv::Mat(3, 3, CV_32F, rotation_mat);

    D = cv::Mat::zeros(4, 1, CV_32F);
    D.at<float>(0, 0) = params_.k1;
    D.at<float>(1, 0) = params_.k2;
    D.at<float>(2, 0) = params_.p1;
    D.at<float>(3, 0) = params_.p2;

    mapx = cv::Mat(h_, w_, CV_32F);
    mapy = cv::Mat(h_, w_, CV_32F);

    camK.at<float>(0, 2) = params_.cx;
    camK.at<float>(1, 2) = params_.cy;
    camK.at<float>(0, 0) = params_.fx;
    camK.at<float>(1, 1) = params_.fy;
    camK.at<float>(2, 2) = 1;
    newCamK = camK.clone();
    invCamK = (newCamK * R.t()).inv(cv::DECOMP_LU);

    detect_filed_ = false;
    d_w_h_ = 0.3;
    localization_ = false;
    can_see_post_ = false;
    letterbox_scale_ = 1.0f;
    letterbox_pad_x_ = 0;
    letterbox_pad_y_ = 0;
}

Vision::~Vision()
{
    LOG(LOG_INFO) << setw(12) << "algorithm:" << setw(18) << "[vision]"
                  << " ended!" << endll;
}

void Vision::set_camera_info(const camera_info &para)
{
    for (auto &item : camera_infos_)
    {
        if (item.second.id == para.id)
        {
            item.second.value = para.value;
            break;
        }
    }
}

Vector2d Vision::odometry(const Vector2i &pos, const robot_math::TransformMatrix &camera_matrix)
{
    float Xw = 0.0, Yw = 0.0;
    float OC = camera_matrix.p().z();

    float roll = -static_cast<float>(camera_matrix.x_rotate());
    float theta = static_cast<float>(camera_matrix.y_rotate());

    // LOG(LOG_INFO)<<OC<<' '<<rad2deg(roll)<<' '<<rad2deg(theta)<<endll;
    roll = roll + odometry_offset_.x();
    theta = theta + odometry_offset_.y();

    Vector2i Pos(pos.x() - params_.cx, params_.cy - pos.y());
    Vector2i calCenterPos(Pos.x() / cos(roll), Pos.y() + Pos.x() * tan(roll));
    Vector2i calPos(calCenterPos.x() + params_.cx, params_.cy - calCenterPos.y());
    double gama = atan((params_.cy - calPos.y()) / params_.fy);
    double O_C_P = M_PI_2 - theta + gama;
    if (!float_equals(O_C_P, M_PI_2))
    {
        Yw = OC * tan(O_C_P);
        Xw = (calPos.x() - params_.cx) * OC * cos(gama) / (cos(O_C_P) * params_.fx);
    }
    return Vector2d(Xw, Yw);
}

Vector2d Vision::camera2self(const Vector2d &pos, double head_yaw)
{
    return rotation_mat_2d(head_yaw + 90.0) * pos;
}

void Vision::get_point_dis(int x, int y)
{
    TransformMatrix camera_matrix;
    frame_mtx_.lock();
    camera_matrix = camera_matrix_;
    frame_mtx_.unlock();
    Vector2d dis(0.0, 0.0);
    dis = odometry(Vector2i(x, y), camera_matrix);
    float xx = static_cast<float>(dis[0]), yy = static_cast<float>(dis[1]);
    tcp_command cmd;
    cmd.type = REMOTE_DATA;
    cmd.size = 2 * enum_size + 2 * float_size;
    remote_data_type t1 = IMAGE_SEND_TYPE;
    image_send_type t2 = IMAGE_SEND_DIS;
    cmd.data.clear();
    cmd.data.append((char *)&t1, enum_size);
    cmd.data.append((char *)&t2, enum_size);
    cmd.data.append((char *)&xx, float_size);
    cmd.data.append((char *)&yy, float_size);
    SERVER->write(cmd);
}

void Vision::run()
{
    if (is_alive_)
    {
        if (camera_src_ == nullptr)
            return;
        if (is_busy_)
            return;
        is_busy_ = true;
        p_count_++;
        robot_math::TransformMatrix camera_matrix;
        float head_yaw, head_pitch;

        cudaError_t err;
        frame_mtx_.lock();
        err = cudaMemcpy(dev_src_, camera_src_, src_size_, cudaMemcpyHostToDevice);
        check_error(err);
        if (OPTS->use_robot())
        {
            camera_matrix = camera_matrix_;
            head_yaw = head_yaw_;
            head_pitch = head_pitch_;
        }
        frame_mtx_.unlock();

        // double t1 = clock();
        const int expected_yuyv = camera_w_ * camera_h_ * 2;
        const int expected_bgr = camera_w_ * camera_h_ * 3;

        if (use_mv_)
        {
            cudaBayer2BGR(dev_src_, dev_bgr_, camera_w_, camera_h_, camera_infos_["saturation"].value,
                          camera_infos_["red_gain"].value, camera_infos_["green_gain"].value, camera_infos_["blue_gain"].value);
        }
        else
        {
            if (src_size_ == expected_yuyv)
            {
                cudaYUYV2BGR(dev_src_, dev_bgr_, camera_w_, camera_h_);
            }
            else if (src_size_ == expected_bgr)
            {
                // ZED-mini backend: buffer already contains BGR uint8 HWC packed.
                err = cudaMemcpy(dev_bgr_, dev_src_, bgr_size_, cudaMemcpyDeviceToDevice);
                check_error(err);
            }
            else
            {
                LOG(LOG_ERROR) << "Vision: unknown camera input size: src_size=" << src_size_
                                << ", expected_yuyv=" << expected_yuyv << ", expected_bgr=" << expected_bgr << endll;
                is_busy_ = false;
                return;
            }
        }
        cudaResizePacked(dev_bgr_, camera_w_, camera_h_, dev_ori_, w_, h_);
        if (use_mv_)
        {
            // MV Bayer path keeps legacy undistortion behavior.
            imgproc::cudaUndistored(dev_ori_, dev_undis_, pCamKData, pDistortData, pInvNewCamKData, pMapxData, pMapyData, w_, h_, 3);
        }
        else if (use_zed_)
        {
            // ZED path: now enable direct undistortion using ZED camera parameters
            imgproc::cudaUndistort(dev_ori_, dev_undis_, 
                                   params_.fx, params_.fy, params_.cx, params_.cy,
                                   params_.k1, params_.k2, 0.0f, params_.p1, params_.p2,
                                   w_, h_, 3);
        }
        else
        {
            // V4L2 YUYV: keep legacy behavior (no undistort).
            cudaMemcpy(dev_undis_, dev_ori_, ori_size_, cudaMemcpyDeviceToDevice);
        }
        // cudaBGR2YUV422(dev_undis_, dev_yuyv_, w_, h_);
        const int in_w = detector_.input_w();
        const int in_h = detector_.input_h();
        cudaResizeLetterbox(dev_undis_, w_, h_, dev_sized_, in_w, in_h,
                    letterbox_scale_, letterbox_pad_x_, letterbox_pad_y_);
        cudaBGR2RGBfp(dev_sized_, dev_rgbfp_, in_w, in_h); // 转为浮点型供神经网络使用

        if (!detector_.detect(dev_rgbfp_,
                               w_, h_,
                               ball_dets_,
                               post_dets_,
                               ball_prob_,
                               post_prob_,
                               min_ball_w_,
                               min_ball_h_,
                               min_post_w_,
                               min_post_h_,
                               d_w_h_,
                               letterbox_scale_,
                               letterbox_pad_x_,
                               letterbox_pad_y_))
        {
            ball_dets_.clear();
            post_dets_.clear();
        }
        // double t2 = clock();
        // LOG(LOG_INFO)<<(t2-t1)/CLOCKS_PER_SEC<<endll;

        if (OPTS->use_robot())
        {
            self_block p = WM->self();
            if (!ball_dets_.empty())
            {
                Vector2i ball_pix(ball_dets_[0].x + ball_dets_[0].w / 2, ball_dets_[0].y + ball_dets_[0].h);
                // ball_pix = undistored(ball_pix);
                Vector2d odo_res = odometry(ball_pix, camera_matrix);
                // LOG(LOG_INFO)<<odo_res.norm()<<endll;
                Vector2d ball_pos = camera2self(odo_res, head_yaw);
                cant_see_ball_count_ = 0;
                Vector2d temp_ball = p.global + rotation_mat_2d(-p.dir) * ball_pos;
                float alpha = (ball_pix.x() - params_.cx) / (float)w_;
                float beta = (ball_pix.y() - params_.cy) / (float)h_;
                WM->set_ball_pos(temp_ball, ball_pos, ball_pix, alpha, beta, true);
            }
            else
            {
                cant_see_ball_count_++;
                if (cant_see_ball_count_ * period_ms_ > 500)
                    WM->set_ball_pos(Vector2d(0, 0), Vector2d(0, 0), Vector2i(0, 0), 0, 0, false);
            }

            if (localization_)
            {
                int post_num = 0;
                vector<goal_post> posts_;
                for (auto &post : post_dets_)
                {
                    dir = imu_data_vison.yaw;
                    post_num++;
                    if (post_num > 2)
                        break;
                    goal_post temp;
                    Vector2i post_pix(post.x + post.w / 2, post.y + post.h * 0.8);
                    Vector2d odo_res = odometry(post_pix, camera_matrix);
                    temp._distance = odo_res.norm() * 100;
                    LOG(LOG_INFO) << "_distance:" << temp._distance << endl;
                    if (temp._distance > 600)
                        continue;
                    Vector2d post_pos = camera2self(odo_res, head_yaw);
                    temp._theta = azimuth_deg(post_pos);
                    posts_.push_back(temp);
                    int id_yaw = ROBOT->get_joint("jhead1")->jid_;
                    int id_pitch = ROBOT->get_joint("jhead2")->jid_;
                    if (posts_.size() == 2)
                    {
                        if (post_dets_[0].x < post_dets_[1].x)
                        {
                            posts_[0].type = goal_post::SENSORMODEL_POST_L;
                            posts_[1].type = goal_post::SENSORMODEL_POST_R;
                        }
                        else
                        {
                            posts_[0].type = goal_post::SENSORMODEL_POST_R;
                            posts_[1].type = goal_post::SENSORMODEL_POST_L;
                        }
                        break;
                    }
                    // if (posts_.size() == 1) // imu判断
                    // {
                    //     if (dir < 0 && id_yaw > 0)
                    //     {
                    //         posts_[0].type = goal_post::SENSORMODEL_POST_R;
                    //     }
                    //     else if (dir > 0 && id_yaw < 0)
                    //     {
                    //         posts_[0].type = goal_post::SENSORMODEL_POST_L;
                    //     }
                    //     LOG(LOG_INFO) << "dir:" << dir << endl;
                    //     LOG(LOG_INFO) << "POST_TYPE:" << posts_[0].type << endl;
                    // }
                }

                LOG(LOG_INFO) << "POST_SIZE:" << post_num << endl;
                if (posts_.size() > 0)
                {
                    can_see_post_count_++;
                    SL->update(player_info(p.global.x(), p.global.y(), p.dir), posts_);
                    if (can_see_post_count_ > 10)
                    {
                        can_see_post_ = true;
                        localization_ = false;
                    }
                }
            }
            else
            {
                can_see_post_ = false;
                can_see_post_count_ = 0;
            }
        }

        if (OPTS->use_debug())
        {
            Mat bgr(h_, w_, CV_8UC3);

            if (OPTS->image_record())
            {
                err = cudaMemcpy(bgr.data, dev_undis_, ori_size_, cudaMemcpyDeviceToHost);
                check_error(err);
                send_image(bgr);
                is_busy_ = false;
                return;
            }
            err = cudaMemcpy(bgr.data, dev_undis_, ori_size_, cudaMemcpyDeviceToHost);
            check_error(err);
            if (img_sd_type_ == IMAGE_SEND_ORIGIN)
            {
                send_image(bgr);
            }
            else if (img_sd_type_ == IMAGE_SEND_RESULT)
            {
                if (detect_filed_)
                {
                    /*
                    for(int j=0; j<h_; j++)
                    {
                        for(int i=0; i<w_; i++)
                        {
                            if(detector_->isGreen(i, j))
                                bgr.at<Vec3b>(j, i) = Vec3b(0, 255, 0);
                        }
                    }*/
                    /*
                    for(int i=0;i<w_;i++)
                        bgr.at<Vec3b>(fieldBorders[i], i) = Vec3b(0, 255, 255);
                        */
                }
                if (!ball_dets_.empty())
                {
                    Vector2i ball_pix(ball_dets_[0].x + ball_dets_[0].w / 2, ball_dets_[0].y + ball_dets_[0].h * 0.95);
                    Vector2d odo_res = odometry(ball_pix, camera_matrix);
                    rectangle(bgr, Point(ball_dets_[0].x, ball_dets_[0].y), Point(ball_dets_[0].x + ball_dets_[0].w, ball_dets_[0].y + ball_dets_[0].h), Scalar(255, 0, 0), 2);
                    // putText(bgr, to_string(odo_res.norm()).substr(0,4), Point(ball_dets_[0].x, ball_dets_[0].y+ball_dets_[0].h),
                    //     FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, 8);
                }
                int i = 0;
                for (auto &dd : post_dets_)
                {
                    if (i >= 2)
                        break;
                    Vector2i post_pix(dd.x + dd.w / 2, dd.y + dd.h * 0.95);
                    Vector2d odo_res = odometry(post_pix, camera_matrix);
                    rectangle(bgr, Point(post_dets_[i].x, post_dets_[i].y), Point(post_dets_[i].x + post_dets_[i].w, post_dets_[i].y + post_dets_[i].h), Scalar(0, 0, 255), 2);
                    // putText(bgr, to_string(odo_res.norm()).substr(0,4), Point(post_dets_[i].x, post_dets_[i].y+post_dets_[i].h),
                    //     FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, 8);
                    i++;
                }
                send_image(bgr);
            }
        }
        is_busy_ = false;
    }
}

void Vision::send_image(const cv::Mat &src)
{
    cv::Mat bgr;
    src.copyTo(bgr);
    std::vector<unsigned char> jpgbuf;
    cv::imencode(".jpg", bgr, jpgbuf);
    bgr.release();
    tcp_command cmd;
    cmd.type = IMG_DATA;
    cmd.size = jpgbuf.size();
    cmd.data.assign((char *)&(jpgbuf[0]), jpgbuf.size());
    SERVER->write(cmd);
}

void Vision::updata(const pub_ptr &pub, const int &type)
{

    if (!is_alive_)
    {
        return;
    }
    if (type == Sensor::SENSOR_CAMERA)
    {
        shared_ptr<Camera> sptr = dynamic_pointer_cast<Camera>(pub);
        if (camera_src_ == nullptr)
        {
            camera_w_ = sptr->camera_w();
            camera_h_ = sptr->camera_h();
            camera_size_ = sptr->camera_size();
            use_mv_ = sptr->use_mv();
            use_zed_ = sptr->use_zed();
            src_size_ = camera_size_;
            bgr_size_ = camera_w_ * camera_h_ * 3;
            camera_src_ = (unsigned char *)malloc(camera_size_);
            cudaError_t err;
            err = cudaMalloc((void **)&dev_src_, src_size_);
            check_error(err);
            err = cudaMalloc((void **)&dev_bgr_, bgr_size_);
            check_error(err);

            if (use_zed_)
            {
                LOG(LOG_WARN) << "Vision: Stage-A ZED path bypasses undistortion to recover debug image visibility." << endll;
            }
        }

        if (!zed_cam_param_applied_ && sptr->use_zed())
        {
            camera_param zed_param;
            if (sptr->get_zed_left_camera_param(zed_param))
            {
                params_ = zed_param;

                D.at<float>(0, 0) = params_.k1;
                D.at<float>(1, 0) = params_.k2;
                D.at<float>(2, 0) = params_.p1;
                D.at<float>(3, 0) = params_.p2;

                camK.at<float>(0, 2) = params_.cx;
                camK.at<float>(1, 2) = params_.cy;
                camK.at<float>(0, 0) = params_.fx;
                camK.at<float>(1, 1) = params_.fy;
                camK.at<float>(2, 2) = 1.0f;
                newCamK = camK.clone();
                invCamK = (newCamK * R.t()).inv(cv::DECOMP_LU);

                if (pCamKData && pInvNewCamKData && pDistortData && pMapxData && pMapyData)
                {
                    cudaError_t err;
                    err = cudaMemcpy(pCamKData, camK.data, 9 * sizeof(float), cudaMemcpyHostToDevice);
                    check_error(err);
                    err = cudaMemcpy(pInvNewCamKData, invCamK.data, 9 * sizeof(float), cudaMemcpyHostToDevice);
                    check_error(err);
                    err = cudaMemcpy(pDistortData, D.data, 4 * sizeof(float), cudaMemcpyHostToDevice);
                    check_error(err);
                    err = cudaMemcpy(pMapxData, mapx.data, h_ * w_ * sizeof(float), cudaMemcpyHostToDevice);
                    check_error(err);
                    err = cudaMemcpy(pMapyData, mapy.data, h_ * w_ * sizeof(float), cudaMemcpyHostToDevice);
                    check_error(err);
                }

                zed_cam_param_applied_ = true;
                LOG(LOG_INFO) << "Vision: applied ZED LEFT calibration fx=" << params_.fx
                              << " fy=" << params_.fy
                              << " cx=" << params_.cx
                              << " cy=" << params_.cy
                              << " k1=" << params_.k1
                              << " k2=" << params_.k2
                              << " p1=" << params_.p1
                              << " p2=" << params_.p2 << endll;
            }
            else
            {
                LOG(LOG_WARN) << "Vision: ZED backend active but calibration not available, keep config camera params." << endll;
            }
        }

        frame_mtx_.lock();
        memcpy(camera_src_, sptr->buffer(), src_size_);
        if (OPTS->use_robot())
        {
            imu_mtx_.lock();
            if (!imu_datas_.empty())
            {
                Imu::imu_data imu_data_ = imu_datas_.front();
                int spf = spfs_.front();
                std::vector<double> foot_degs = foot_degs_.front();
                std::vector<double> head_degs = head_degs_.front();
                AngleAxisd roll, pitch;
                pitch = AngleAxisd(deg2rad(imu_data_.pitch), Vector3d::UnitY());
                roll = AngleAxisd(deg2rad(imu_data_.roll), Vector3d::UnitX());
                Quaternion<double> quat = roll * pitch;

                head_yaw_ = -head_degs[0];
                head_pitch_ = head_degs[1];
                TransformMatrix body = ROBOT->leg_forward_kinematics(foot_degs, spf);
                body.set_R(quat.matrix());
                camera_matrix_ = body * TransformMatrix(0, 0, ROBOT->trunk_length()) * TransformMatrix(head_degs[0], 'z') * TransformMatrix(0, 0, ROBOT->neck_length()) * TransformMatrix(head_degs[1], 'y') * TransformMatrix(-0.02, 0, ROBOT->head_length());
            }
            imu_mtx_.unlock();
        }
        frame_mtx_.unlock();
        return;
    }
    if (type == Sensor::SENSOR_IMU)
    {
        shared_ptr<Imu> sptr = dynamic_pointer_cast<Imu>(pub);
        imu_mtx_.lock();
        imu_datas_.push(sptr->data());
        if (imu_datas_.size() > 4)
            imu_datas_.pop();
        int spf = WM->support_foot();
        foot_degs_.push(ROBOT->get_foot_degs(spf));
        head_degs_.push(ROBOT->get_head_degs());
        spfs_.push(spf);
        if (foot_degs_.size() > 4)
            foot_degs_.pop();
        if (head_degs_.size() > 4)
            head_degs_.pop();
        if (spfs_.size() > 4)
            spfs_.pop();
        imu_mtx_.unlock();
        return;
    }
}

bool Vision::start()
{
    std::string engine_path = CONF->get_config_value<string>("net_engine_file");
    if (!detector_.load(engine_path, ball_id_, post_id_))
    {
        LOG(LOG_ERROR) << "Vision: failed to load TensorRT engine: " << engine_path << endll;
        return false;
    }

    ori_size_ = w_ * h_ * 3;
    yuyv_size_ = w_ * h_ * 2;
    const int in_w = detector_.input_w();
    const int in_h = detector_.input_h();
    sized_size_ = in_w * in_h * 3;
    rgbf_size_ = in_w * in_h * 3 * sizeof(float);

    cudaError_t err;
    err = cudaMalloc((void **)&dev_ori_, ori_size_);
    check_error(err);
    err = cudaMalloc((void **)&dev_undis_, ori_size_);
    check_error(err);
    err = cudaMalloc((void **)&dev_yuyv_, yuyv_size_);
    check_error(err);
    err = cudaMalloc((void **)&dev_sized_, sized_size_);
    check_error(err);
    err = cudaMalloc((void **)&dev_rgbfp_, rgbf_size_);
    check_error(err);

    err = cudaMalloc(&pCamKData, 9 * sizeof(float));
    check_error(err);
    err = cudaMalloc(&pInvNewCamKData, 9 * sizeof(float));
    check_error(err);
    err = cudaMalloc(&pDistortData, 4 * sizeof(float));
    check_error(err);
    err = cudaMalloc(&pMapxData, h_ * w_ * sizeof(float));
    check_error(err);
    err = cudaMalloc(&pMapyData, h_ * w_ * sizeof(float));
    check_error(err);

    err = cudaMemcpy(pCamKData, camK.data, 9 * sizeof(float), cudaMemcpyHostToDevice);
    check_error(err);
    err = cudaMemcpy(pInvNewCamKData, invCamK.data, 9 * sizeof(float), cudaMemcpyHostToDevice);
    check_error(err);
    err = cudaMemcpy(pDistortData, D.data, 4 * sizeof(float), cudaMemcpyHostToDevice);
    check_error(err);
    err = cudaMemcpy(pMapxData, mapx.data, h_ * w_ * sizeof(float), cudaMemcpyHostToDevice);
    check_error(err);
    err = cudaMemcpy(pMapyData, mapy.data, h_ * w_ * sizeof(float), cudaMemcpyHostToDevice);
    check_error(err);

    is_alive_ = true;
    start_timer();
    return true;
}

void Vision::stop()
{
    if (is_alive_)
    {
        delete_timer();
        detector_.release();
        free(camera_src_);
        cudaFree(dev_ori_);
        cudaFree(dev_undis_);
        cudaFree(dev_yuyv_);
        cudaFree(dev_src_);
        cudaFree(dev_bgr_);
        cudaFree(dev_rgbfp_);
        cudaFree(dev_sized_);

        cudaFree(pCamKData);
        cudaFree(pInvNewCamKData);
        cudaFree(pDistortData);
        cudaFree(pMapxData);
        cudaFree(pMapyData);
    }
    is_alive_ = false;
}
