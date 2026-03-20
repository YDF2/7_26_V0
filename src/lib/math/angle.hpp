#ifndef __ANGLE_HPP
#define __ANGLE_HPP

#include <eigen3/Eigen/Dense>
#include <cmath>

namespace robot_math
{
    inline double deg2rad(const double &x)
    {
        return x * M_PI / 180.0;
    }

    inline double rad2deg(const double &x)
    {
        return x * 180.0 / M_PI;
    }

    inline double cos_deg(const double &d)
    {
        return cos(deg2rad(d));
    }

    inline double sin_deg(const double &d)
    {
        return sin(deg2rad(d));
    }

    inline double tan_deg(const double &d)
    {
        return tan(deg2rad(d));
    }

    template<typename T>
    inline double azimuth_deg(const T &v)
    {
        return rad2deg(std::atan2(v.y(), v.x()));
    }

    template<typename T>
    inline double azimuth_rad(const T &v)
    {
        return std::atan2(v.y(), v.x());
    }

    template<typename T>
    inline T normalize_deg(T deg)
    {
        while (deg > 180.0)
            deg -= 360.0;
        while (deg < -180.0)
            deg += 360.0;
        return deg;
    }

    template<typename T>
    inline T normalize_rad(T rad)
    {
        while (rad > M_PI)
            rad -= 2*M_PI;
        while (rad < -M_PI)
            rad += 2*M_PI;
        return rad;
    }
}

#endif
