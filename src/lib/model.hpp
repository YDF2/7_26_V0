#ifndef __MODEL_HPP
#define __MODEL_HPP

#include <string>
#include <map>
#include <map>
#include <eigen3/Eigen/Dense>

struct goal_post
{
    enum post_type
    {
        SENSORMODEL_POST_UNKNOWN,
        SENSORMODEL_POST_L, //left post
        SENSORMODEL_POST_R, //right post
    };

    enum post_side
    {
        SENSORMODEL_SIDE_OUR,
        SENSORMODEL_SIDE_OPP,
        SENSORMODEL_SIDE_UNKNOW,
    };

    post_type type = SENSORMODEL_POST_UNKNOWN;
    post_side side = SENSORMODEL_SIDE_UNKNOW;
    float _theta = 0.0f;
    float _distance = 1000.0f;
};

struct filed_info
{
    int field_length;
    int field_width;
    int goal_depth;
    int goal_width;
    int goal_height;
    int goal_area_length;
    int goal_area_width;
    int penalty_mark_distance;
    int center_circle_diameter;
    int border_strip_width_min;

    float scale=1.0;

    void scale_field(float s=1.0)
    {
        scale *= s;
        field_length *= s;
        field_width *= s;
        goal_depth *= s;
        goal_width *= s;
        goal_height *= s;;
        goal_area_length *= s;
        goal_area_width *= s;
        penalty_mark_distance *= s;
        center_circle_diameter *= s;
        border_strip_width_min *= s;
    }
};

struct player_info
{
    int id;
    float x, y, dir;
    float ball_x, ball_y;
    bool can_see = false;
    bool my_kick = false;
    player_info(float m_x, float m_y, float m_d)
        :x(m_x), y(m_y), dir(m_d){}

    player_info(){}
};

struct ball_block
{
    Eigen::Vector2d global;
    Eigen::Vector2d self;
    Eigen::Vector2i pixel;
    float alpha, beta; // alpha：左负右正；beta：上小下大
    bool can_see=false;
};

struct self_block
{
    Eigen::Vector2d global;
    double dir;
    bool sure=false;
};

struct camera_info
{
    std::string name;
    int id;
    float value;
    float default_value;
    float min_value;
    float max_value;
};

struct camera_param
{
    float fx, fy;
    float cx, cy;
    float k1, k2;
    float p1, p2;
};

struct object_det
{
    int id;
    float prob;
    int x, y, w, h;
    object_det(int i=0, float p=1, int x_=0, int y_=0, int w_=0, int h_=0) 
        : id(i), prob(p), x(x_), y(y_), w(w_), h(h_){}
    bool operator< (const object_det &obj)
    {
        return prob<obj.prob;
    }
};

enum fsm_state
{
    FSM_STATE_READY,
    FSM_STATE_GETUP,
    FSM_STATE_SEARCH_BALL,
    FSM_STATE_GOTO_BALL,
    FSM_STATE_KICK_BALL,
    FSM_STATE_DRIBBLE,
    FSM_STATE_SL
};

enum { player_info_size = sizeof(player_info)};

#endif
