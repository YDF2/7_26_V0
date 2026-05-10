#include <ctime>
#include <stdlib.h>
#include <limits.h>
#include "localization.h"
#include "configuration.hpp"
#include "core/worldmodel.hpp"

using namespace std;

int flag;
Localization::Localization() : _haslocated(false)
{
  srand(time(0));

  //_conf._sample_num = 60;
  _conf._div_dir = 36;
  _conf._observation_update_counter = 1;
  _kalman.init();
}

Localization::~Localization()
{
}

bool Localization::update(const player_info &player_info_, const vector<goal_post> &posts_)
{
  clock_t start = clock();
  _kalman.forecast(player_info_);
  KA::State state;

  static int obser_count = 0;
  obser_count++;

  if (obser_count % _conf._observation_update_counter == 0)
  {
    // bool check=_kalman.fieldLineUpdate(sensormodel.fieldlines);
    int num = _kalman.goalPostUpdate(posts_);

    if (flag != num)
      _kalman.setPzero();
    if (num == 1)
    {
      state = _kalman.obeupdate1();
    }
    else if (num == 2)
    {
      state = _kalman.obeupdate2();
    }
    else if (num == 0)
    {
      state.x = player_info_.x * 100.0;
      state.y = player_info_.y * 100.0;
    }

    flag = num;
  }
  LOG(LOG_INFO) << "state.x" << state.x / 100.0 << '\t' << "state.y" << state.y / 100.0 << endll;
  if (abs(state.x) <= SOCCERMAP->width() / 2 && abs(state.y) <= SOCCERMAP->height())
    WM->set_my_pos(Eigen::Vector2d(state.x / 100.0, state.y / 100.0));
  else
  {
    LOG(LOG_INFO) << "out" << endll;
    _kalman.setPzero();
  }

  clock_t finish = clock();

  _haslocated = true;
  return true;
}
