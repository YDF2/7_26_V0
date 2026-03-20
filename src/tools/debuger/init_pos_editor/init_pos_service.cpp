#include "init_pos_service.hpp"
#include <QFile>
#include <QRegularExpression>

bool InitPosService::parse_strategy_range(const QString &text, int &begin, int &end) const
{
    begin = text.indexOf("\"strategy\"");
    if (begin < 0)
        return false;
    begin = text.indexOf('{', begin);
    if (begin < 0)
        return false;
    int depth = 0;
    for (int i = begin; i < text.size(); ++i)
    {
        QChar ch = text[i];
        if (ch == '{')
            depth++;
        else if (ch == '}')
        {
            depth--;
            if (depth == 0)
            {
                end = i;
                return true;
            }
        }
    }
    return false;
}

QString InitPosService::format_number(double value) const
{
    QString txt = QString::number(value, 'f', 3);
    while (txt.contains('.') && txt.endsWith('0'))
        txt.chop(1);
    if (txt.endsWith('.'))
        txt.append('0');
    return txt;
}

bool InitPosService::load_by_player_id(int player_id, InitPosInfo &info, QString &error_msg) const
{
    QFile file("data/config.conf");
    if (!file.open(QIODevice::ReadOnly))
    {
        error_msg = "Cannot open data/config.conf";
        return false;
    }
    QString text = QString::fromUtf8(file.readAll());
    file.close();
    int strategy_begin = -1, strategy_end = -1;
    if (!parse_strategy_range(text, strategy_begin, strategy_end))
    {
        error_msg = "Strategy section not found";
        return false;
    }
    QString strategy_text = text.mid(strategy_begin, strategy_end - strategy_begin + 1);
    QRegularExpression role_re("\"([A-Za-z_][A-Za-z0-9_]*)\"\\s*:\\s*\\{([\\s\\S]*?)\\n\\s*\\}");
    QRegularExpression id_re("\"id\"\\s*:\\s*(-?\\d+)");
    QRegularExpression init_re("\"init_pos\"\\s*:\\s*\\[\\s*([-+]?\\d*\\.?\\d+)\\s*,\\s*([-+]?\\d*\\.?\\d+)\\s*\\]");
    auto role_it = role_re.globalMatch(strategy_text);
    while (role_it.hasNext())
    {
        auto match = role_it.next();
        QString role_name = match.captured(1);
        QString role_body = match.captured(2);
        auto id_match = id_re.match(role_body);
        if (!id_match.hasMatch())
            continue;
        if (id_match.captured(1).toInt() != player_id)
            continue;
        auto init_match = init_re.match(role_body);
        if (!init_match.hasMatch())
        {
            error_msg = "init_pos field not found";
            return false;
        }
        info.player_id = player_id;
        info.role = role_name;
        info.x = init_match.captured(1).toDouble();
        info.y = init_match.captured(2).toDouble();
        return true;
    }
    error_msg = "Role not found by player id";
    return false;
}

bool InitPosService::update_by_player_id(int player_id, double x, double y, QString &error_msg) const
{
    QFile file("data/config.conf");
    if (!file.open(QIODevice::ReadOnly))
    {
        error_msg = "Cannot open data/config.conf";
        return false;
    }
    QString text = QString::fromUtf8(file.readAll());
    file.close();
    int strategy_begin = -1, strategy_end = -1;
    if (!parse_strategy_range(text, strategy_begin, strategy_end))
    {
        error_msg = "Strategy section not found";
        return false;
    }
    QString strategy_text = text.mid(strategy_begin, strategy_end - strategy_begin + 1);
    QRegularExpression role_re("\"([A-Za-z_][A-Za-z0-9_]*)\"\\s*:\\s*\\{([\\s\\S]*?)\\n\\s*\\}");
    QRegularExpression id_re("\"id\"\\s*:\\s*(-?\\d+)");
    QRegularExpression init_re("(\"init_pos\"\\s*:\\s*\\[\\s*)([-+]?\\d*\\.?\\d+)(\\s*,\\s*)([-+]?\\d*\\.?\\d+)(\\s*\\])");
    auto role_it = role_re.globalMatch(strategy_text);
    int role_begin = -1;
    int role_len = -1;
    QString role_text;
    while (role_it.hasNext())
    {
        auto match = role_it.next();
        QString role_body = match.captured(2);
        auto id_match = id_re.match(role_body);
        if (!id_match.hasMatch())
            continue;
        if (id_match.captured(1).toInt() == player_id)
        {
            role_begin = match.capturedStart(0);
            role_len = match.capturedLength(0);
            role_text = match.captured(0);
            break;
        }
    }
    if (role_begin < 0)
    {
        error_msg = "Role not found by player id";
        return false;
    }
    auto init_match = init_re.match(role_text);
    if (!init_match.hasMatch())
    {
        error_msg = "init_pos field not found";
        return false;
    }
    QString new_init = init_match.captured(1) + format_number(x) + init_match.captured(3) + format_number(y) + init_match.captured(5);
    role_text.replace(init_match.capturedStart(0), init_match.capturedLength(0), new_init);
    strategy_text.replace(role_begin, role_len, role_text);
    text.replace(strategy_begin, strategy_end - strategy_begin + 1, strategy_text);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        error_msg = "Cannot write data/config.conf";
        return false;
    }
    file.write(text.toUtf8());
    file.close();
    return true;
}
