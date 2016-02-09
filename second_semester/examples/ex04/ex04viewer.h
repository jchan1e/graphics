#ifndef EX04V_H
#define EX04V_H

#include <QWidget>
#include <QSlider>
#include <QDoubleSpinBox>
#include "ex04opengl.h"

class Ex04viewer : public QWidget
{
Q_OBJECT
private:
   QSlider*     Lpos;
   QSlider*     Zpos;
   QPushButton* light;
   Ex04opengl*  ogl;
private slots:
   void lmove();        //  Light movement
public:
    Ex04viewer();
};

#endif
