#include "init_pos_form.hpp"

InitPosForm::InitPosForm()
{
    role_lab_ = new QLabel("-");
    x_spin_ = new QDoubleSpinBox();
    y_spin_ = new QDoubleSpinBox();
    save_btn_ = new QPushButton("Save");
    x_spin_->setRange(-10.0, 10.0);
    y_spin_->setRange(-10.0, 10.0);
    x_spin_->setSingleStep(0.05);
    y_spin_->setSingleStep(0.05);
    x_spin_->setDecimals(3);
    y_spin_->setDecimals(3);
    QGridLayout *layout = new QGridLayout;
    layout->addWidget(new QLabel("Role:"), 0, 0);
    layout->addWidget(role_lab_, 0, 1, 1, 2);
    layout->addWidget(new QLabel("X:"), 1, 0);
    layout->addWidget(x_spin_, 1, 1, 1, 2);
    layout->addWidget(new QLabel("Y:"), 2, 0);
    layout->addWidget(y_spin_, 2, 1, 1, 2);
    layout->addWidget(save_btn_, 3, 1, 1, 2);
    setLayout(layout);
}

void InitPosForm::set_role(const QString &role)
{
    role_lab_->setText(role);
}

void InitPosForm::set_values(double x, double y)
{
    x_spin_->setValue(x);
    y_spin_->setValue(y);
}

double InitPosForm::x() const
{
    return x_spin_->value();
}

double InitPosForm::y() const
{
    return y_spin_->value();
}

QPushButton *InitPosForm::save_button() const
{
    return save_btn_;
}

QLabel *InitPosForm::role_label() const
{
    return role_lab_;
}
