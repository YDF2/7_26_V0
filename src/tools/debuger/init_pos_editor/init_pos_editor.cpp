#include "init_pos_editor.hpp"

InitPosEditor::InitPosEditor(int player_id, QWidget *parent): player_id_(player_id), QMainWindow(parent)
{
    setAttribute(Qt::WA_DeleteOnClose);
    form_ = new InitPosForm();
    setCentralWidget(form_);
    setWindowTitle("Init Position Editor");
    connect(form_->save_button(), &QPushButton::clicked, this, &InitPosEditor::procSave);
    loadData();
}

void InitPosEditor::loadData()
{
    InitPosInfo info;
    QString error_msg;
    if (!service_.load_by_player_id(player_id_, info, error_msg))
    {
        statusBar()->showMessage(error_msg, 3000);
        form_->set_role("N/A");
        return;
    }
    form_->set_role(info.role);
    form_->set_values(info.x, info.y);
    statusBar()->showMessage("Loaded", 2000);
}

void InitPosEditor::procSave()
{
    QString error_msg;
    if (!service_.update_by_player_id(player_id_, form_->x(), form_->y(), error_msg))
    {
        statusBar()->showMessage(error_msg, 3000);
        return;
    }
    statusBar()->showMessage("Saved to data/config.conf", 3000);
}

void InitPosEditor::closeEvent(QCloseEvent *event)
{
    Q_UNUSED(event);
    emit closed();
}
