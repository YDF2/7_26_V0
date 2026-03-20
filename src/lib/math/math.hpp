#ifndef __MATH_HPP
#define __MATH_HPP

#include "matrix.hpp"

namespace robot_math
{
    inline Eigen::Matrix2d rotation_mat_2d(const double &deg)
    {
        Eigen::Matrix2d temp;
        double rad = deg2rad(deg);
        temp<<std::cos(rad), std::sin(rad),
                -std::sin(rad), std::cos(rad);
        return temp;
    }

    template<typename T>
    inline bool float_equals(T v1, T v2)
    {
        return fabs(v1-v2)<1E-5;
    }
}

#endif
