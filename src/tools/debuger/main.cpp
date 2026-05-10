
#include <QApplication>
#include "debuger.hpp"
#include "configuration.hpp"
#include <GL/glut.h>

using namespace std;
using namespace robot;

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    glutInit(&argc, argv);
    Debuger foo;
    foo.show();
    return app.exec();
}
