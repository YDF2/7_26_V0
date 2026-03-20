#ifndef __INIT_POS_EDITOR_HPP
#define __INIT_POS_EDITOR_HPP

#include <QtWidgets>
#include "init_pos_form.hpp"
#include "init_pos_service.hpp"

class InitPosEditor: public QMainWindow
{
    Q_OBJECT
public:
    explicit InitPosEditor(int player_id, QWidget *parent = nullptr);

signals:
    void closed();

protected:
    void closeEvent(QCloseEvent *event);

private slots:
    void procSave();

private:
    void loadData();
    int player_id_;
    InitPosForm *form_;
    InitPosService service_;
};

#endif
