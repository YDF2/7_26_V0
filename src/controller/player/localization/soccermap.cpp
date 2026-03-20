#include "soccermap.h"
#include "parser/parser.hpp"

using namespace std;
using namespace robot_math;
using namespace Eigen;

SoccerMap::SoccerMap()
{
    getConfigInfo();
    _center = Vector2i(_field_length / 2, _field_width / 2);
    _goal_post_position[FLAG_DOOR_OPP_P] = Vector2f(_field_length / 2, _goal_width / 2);
    _goal_post_position[FLAG_DOOR_OPP_N] = Vector2f(_field_length / 2, -_goal_width / 2);
    _goal_post_position[FLAG_DOOR_OUR_P] = Vector2f(-_field_length / 2, _goal_width / 2);
    _goal_post_position[FLAG_DOOR_OUR_N] = Vector2f(-_field_length / 2, -_goal_width / 2);

    _our_goal = Vector2f(-_field_length / 2, 0);
    _opp_goal = Vector2f(_field_length / 2, 0);

}

bool SoccerMap::getConfigInfo()
{
    if(!parser::parse(CONF->field_file(), filed_info_))
    {
        return false;
    }
    _field_length = filed_info_.field_length;
    _field_width = filed_info_.field_width;
    _goal_depth = filed_info_.goal_depth;
    _goal_width = filed_info_.goal_width;
    _goal_height = filed_info_.goal_height;
    _goal_area_length = filed_info_.goal_area_length;
    _goal_area_width = filed_info_.goal_area_width;
    _penalty_mark_distance = filed_info_.penalty_mark_distance;
    _center_circle_diameter = filed_info_.center_circle_diameter;
    _border_strip_width_min = filed_info_.border_strip_width_min;
    return true;
}
