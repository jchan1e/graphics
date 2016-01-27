//
//  OpenGL Ex02opengl Widget
//

#ifndef EX02OPENGL_H
#define EX02OPENGL_H

#include <QtOpenGL>
#include <QString>
#include <QVector>
#include "Object.h"

class Ex02opengl : public QGLWidget
{
Q_OBJECT
private:
   bool    init;      //  Initialized
   int     x0,y0,z0;  //  Object center (percent)
   int     mode;      //  Display mode
   int     th,ph;     //  Display angles
   bool    mouse;     //  Mouse pressed
   QPoint  pos;       //  Mouse position
   int     fov;       //  Field of view
   double  dim;       //  Display size
   double  asp;       //  Sceen aspect ratio
   Object* obj;       //  Object
   QGLShaderProgram shader[4]; //  Shaders
   QVector<Object*> objects;   //  Objects
public:
   Ex02opengl(QWidget* parent=0);                  //  Constructor
   QSize sizeHint() const {return QSize(400,400);} //  Default size of widget
public slots:
    void setShader(int sel);               //  Slot to set shader
    void setPerspective(int on);           //  Slot to set projection type
    void setObject(int type);              //  Slot to set displayed object
    void setX(int X);                      //  Slot to set X position (percent)
    void setY(int Y);                      //  Slot to set Y position (percent)
    void setZ(int Z);                      //  Slot to set Z position (percent)
signals:
    void angles(QString text);             //  Signal for view angles
    void center(QString text);             //  Signal for object center
protected:
    void initializeGL();                   //  Initialize widget
    void resizeGL(int width, int height);  //  Resize widget
    void paintGL();                        //  Draw widget
    void mousePressEvent(QMouseEvent*);    //  Mouse pressed
    void mouseReleaseEvent(QMouseEvent*);  //  Mouse released
    void mouseMoveEvent(QMouseEvent*);     //  Mouse moved
    void wheelEvent(QWheelEvent*);         //  Mouse wheel
private:
   void Fatal(QString message);            //  Error handler
   void Projection();                      //  Update projection
};

#endif
