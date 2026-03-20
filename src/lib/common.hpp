#ifndef __COMMON_HPP
#define __COMMON_HPP

#include <ctime>
#include <cstdio>
#include <string>
#include <iomanip>
#include <unistd.h>

inline std::string get_time()
{
    time_t timep;
    std::time(&timep);
    char tmp[64];
    std::strftime(tmp, sizeof(tmp), "%Y-%m-%d_%H%M%S", std::localtime(&timep));
    return std::string(tmp);
}

inline std::string get_project_dir()
{
    char buf1[256];
	getcwd(buf1, sizeof(buf1)); // get Current Working Directory
    std::string cwd(buf1);
    std::string::size_type pos = cwd.find("/bin");
    if(pos == std::string::npos)
        return "";
    else
        return cwd.substr(0, pos+1);
}

inline std::string get_source_dir()
{
    return get_project_dir()+"src/";
}

#endif
