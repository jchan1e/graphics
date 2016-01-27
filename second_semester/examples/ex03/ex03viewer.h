#ifndef EX03V_H
#define EX03V_H

#include <QWidget>
#include <QSlider>
#include <QDoubleSpinBox>
#include "ex03opengl.h"

class Ex03viewer : public QWidget
{
Q_OBJECT
private:
   QDoubleSpinBox* Xpos;
   QDoubleSpinBox* Ypos;
   QSlider*        Zpos;
   QPushButton*    light;
   Ex03opengl*     ogl;
private slots:
   void reset();        //  Reset angles
   void lmove();        //  Pause/animate light
   void izoom(int iz);  //  Zoom level (percent)
public:
    Ex03viewer();
};

#endif
