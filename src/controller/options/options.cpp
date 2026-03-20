#include "options.hpp"
#include <iostream>
#include "logger.hpp"

using namespace std;
using namespace boost::program_options;

Options::Options(): opts_desc_("  Options description")
{
    // 向option_description对象添加选项
    opts_desc_.add_options()
    ("help,h", "Print this message and exit.")
    ("player,p", value<int>()->default_value(0),
     "Player ID number.")
    ("debug,d", value<bool>()->default_value(false),
     "If you want to start in debug mode.")
    ("camera,c", value<bool>()->default_value(true),
     "If you want to use camera.")
    ("robot,r", value<bool>()->default_value(true),
     "If you want to use robot")
    ("gamectrl,g", value<bool>()->default_value(true),
    "If you want to use game controller")
    ("say,s", value<bool>()->default_value(true),
    "If you want to use communication")
    ("kick,k", value<int>()->default_value(0),
     "Kick mode\n0. Normal kick.\n1. Penalty kick")
    ("mote,m", value<bool>()->default_value(false),
     "If you want to use remote.")
    ("image,i", value<bool>()->default_value(false),
     "If you want to record image.");
}

bool Options::init(int argc, char **argv)
{
    try
    {
        store(parse_command_line(argc, argv, opts_desc_), var_map_); // 解析命令行输入的参数,并存储至variables_map中
        // ? 更新外部变量
        id_ = arg<int>("player");
        use_debug_ = arg<bool>("debug");
        use_camera_ = arg<bool>("camera");
        use_robot_ = arg<bool>("robot");
        use_remote_ = arg<bool>("mote");
        use_comm_ = arg<bool>("say");
        use_gc_ = arg<bool>("gamectrl");
        image_record_ = arg<bool>("image");
        kick_mode_ = arg<int>("kick");

        if (var_map_.count("help"))
        {
            LOG(LOG_INFO) << '\n' << opts_desc_ << endll;
            return false;
        }
    }
    catch (boost::program_options::unknown_option &e)
    {
        LOG(LOG_ERROR) << e.what() << endll;
        LOG(LOG_INFO) << '\n' << opts_desc_ << endll;
    }
    catch (std::exception &e)
    {
        LOG(LOG_ERROR)<< e.what() << endll;
        LOG(LOG_INFO) << '\n' << opts_desc_ << endll;
        return false;
    }

    return true;
}