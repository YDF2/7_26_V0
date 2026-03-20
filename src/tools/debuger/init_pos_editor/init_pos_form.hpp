#ifndef __INIT_POS_FORM_HPP
#define __INIT_POS_FORM_HPP

#include <QtWidgets>

class InitPosForm: public QWidget
{
    Q_OBJECT
public:
    InitPosForm();
    void set_role(const QString &role);
    void set_values(double x, double y);
    double x() const;
    double y() const;
    QPushButton *save_button() const;
    QLabel *role_label() const;

private:
    QDoubleSpinBox *x_spin_;
    QDoubleSpinBox *y_spin_;
    QLabel *role_lab_;
    QPushButton *save_btn_;
};

#endif
