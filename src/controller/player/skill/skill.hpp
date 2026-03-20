#ifndef __SKILL_HPP
#define __SKILL_HPP

#include "model.hpp"
#include "task/task.hpp"

extern task_ptr skill_goto(const self_block &self, const Eigen::Vector2d &target, double dir);
extern task_ptr skill_penalty_kick(const ball_block &ball);

#endif
