#ifndef __PARSER_HPP
#define __PAESER_HPP

#include <unordered_map>
#include <unordered_set>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include "robot/robot_define.hpp"
#include "class_exception.hpp"
#include "model.hpp"


namespace parser
{
    namespace bpt = boost::property_tree;

    robot::bone_ptr parse(const std::string &filename, robot::bone_map &bone_map, robot::joint_map &joint_map);

    bool parse(const std::string &act_file, std::map<std::string, robot::robot_act> &acts, std::map<std::string, robot::robot_pos> &poses);
    void save(const std::string &act_file, const std::map<std::string, robot::robot_act> &acts, const std::map<std::string, robot::robot_pos> &poses);

    bool parse(const std::string &filename, std::map<std::string, camera_info> &ctrl_infos);
    void save(const std::string &filename, const std::map<std::string, camera_info> &ctrl_infos);

    bool parse(const std::string &filename, camera_param &param);

    bool parse(const std::string &cfgname, bpt::ptree &pt);

    bool parse(const std::string &filename, filed_info &field);

    bool parse(const std::string &filename, robot::joint_map &jmap);
    void save(const std::string &filename, const robot::joint_map &jmap);
    
};

#endif