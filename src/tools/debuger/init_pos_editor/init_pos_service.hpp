#ifndef __INIT_POS_SERVICE_HPP
#define __INIT_POS_SERVICE_HPP

#include <QString>
#include "init_pos_types.hpp"

class InitPosService
{
public:
    bool load_by_player_id(int player_id, InitPosInfo &info, QString &error_msg) const;
    bool update_by_player_id(int player_id, double x, double y, QString &error_msg) const;

private:
    bool parse_strategy_range(const QString &text, int &begin, int &end) const;
    QString format_number(double value) const;
};

#endif
