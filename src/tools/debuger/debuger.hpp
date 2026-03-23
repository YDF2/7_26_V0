#ifndef __DEBUGER_H
#define __DEBUGER_H

#include <QtWidgets>

class Debuger: public QMainWindow
{
    Q_OBJECT
public:
    Debuger();
    QComboBox *pBox;

protected:
    void resizeEvent(QResizeEvent *event) override;

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
    void procBtnIPE();
    void procBtnJR();
private:
    void cacheBaseFonts();
    void applyScale(double scale);

    int opened_debuger_count;
    int current_player;
    QSize base_window_size;
    QHash<QWidget*, QFont> base_fonts;
};

#endif
