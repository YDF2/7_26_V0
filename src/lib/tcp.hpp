#ifndef __TCP_HPP
#define __TCP_HPP

#include <functional>
#include <string>
#include <vector>

#define MAX_CMD_LEN 512

enum tcp_data_dir
{
    DIR_BOTH = 0,
    DIR_APPLY = 1,
    DIR_SUPPLY = 2
};

enum tcp_cmd_type
{
    NONE_DATA = 0,
    REG_DATA = 2,
    TEST_DATA = 5,
    JOINT_DATA = 6,
    REMOTE_DATA = 7,
    IMG_DATA = 8,
    END_DATA = 20
};

struct tcp_command
{
    int timestamp;
    tcp_cmd_type type;
    bool end;
    unsigned int size;
    std::string data;
};

enum remote_data_type
{
    NON_DATA = 0,
    WALK_DATA = 1,
    ACT_DATA = 2,
    LOOKAT_DATA = 3,
    JOINT_OFFSET = 4,
    JOINT_SET = 5,
    CAMERA_SET = 10,
    IMAGE_SEND_TYPE = 12,
};

enum image_send_type
{
    IMAGE_SEND_ORIGIN = 0,
    IMAGE_SEND_RESULT = 1,
    IMAGE_SEND_RECT=2,
    IMAGE_SEND_POINT=3,
    IMAGE_SEND_DIS=4
};

struct remote_data
{
    remote_data_type type;
    unsigned int size;
    std::string data;
};

enum {enum_size = sizeof(tcp_data_dir)};
enum {float_size = sizeof(float)};
enum {int_size = sizeof(int)};
enum {bool_size = sizeof(bool)};
enum {data_offset = int_size + enum_size + bool_size + int_size};
enum {max_data_size = MAX_CMD_LEN - data_offset};

typedef std::function<void (const tcp_command)> tcp_callback;

#endif
