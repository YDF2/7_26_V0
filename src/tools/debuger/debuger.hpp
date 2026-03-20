#ifndef __DEBUGER_H
#define __DEBUGER_H

#include <QtWidgets>

class Debuger: public QMainWindow
{
    Q_OBJECT
public:
    Debuger();
    QComboBox *pBox;

public slots:
    void procPlayerChanged(int index);
    void procClosed();
    void procBtnIM();
    void procBtnID();
    void procBtnCS();
    void procBtnTM();
    void procBtnAD();
    void procBtnAM();
    void procBtnWR();
    void procBtnJR();
private:
    int opened_debuger_count;
    int current_player;
};

#endif
