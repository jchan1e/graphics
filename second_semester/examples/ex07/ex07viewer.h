#ifndef EX07V_H
#define EX07V_H

#include <QWidget>
#include <QSlider>
#include <QDoubleSpinBox>
#include "ex07opengl.h"

class Ex07viewer : public QWidget
{
Q_OBJECT
private:
   QSlider*     Lpos;
   QSlider*     Zpos;
   QPushButton* light;
   Ex07opengl*  ogl;
private slots:
   void lmove();        //  Light movement
public:
    Ex07viewer();
};

#endif
