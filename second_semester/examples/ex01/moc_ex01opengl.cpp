/****************************************************************************
** Meta object code from reading C++ file 'ex01opengl.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.2.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "ex01opengl.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ex01opengl.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.2.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_Ex01opengl_t {
    QByteArrayData data[10];
    char stringdata[80];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    offsetof(qt_meta_stringdata_Ex01opengl_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData) \
    )
static const qt_meta_stringdata_Ex01opengl_t qt_meta_stringdata_Ex01opengl = {
    {
QT_MOC_LITERAL(0, 0, 10),
QT_MOC_LITERAL(1, 11, 6),
QT_MOC_LITERAL(2, 18, 0),
QT_MOC_LITERAL(3, 19, 4),
QT_MOC_LITERAL(4, 24, 9),
QT_MOC_LITERAL(5, 34, 2),
QT_MOC_LITERAL(6, 37, 14),
QT_MOC_LITERAL(7, 52, 9),
QT_MOC_LITERAL(8, 62, 4),
QT_MOC_LITERAL(9, 67, 11)
    },
    "Ex01opengl\0angles\0\0text\0setShader\0on\0"
    "setPerspective\0setObject\0type\0setLighting\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Ex01opengl[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   39,    2, 0x06,

 // slots: name, argc, parameters, tag, flags
       4,    1,   42,    2, 0x0a,
       6,    1,   45,    2, 0x0a,
       7,    1,   48,    2, 0x0a,
       9,    1,   51,    2, 0x0a,

 // signals: parameters
    QMetaType::Void, QMetaType::QString,    3,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    8,
    QMetaType::Void, QMetaType::Int,    5,

       0        // eod
};

void Ex01opengl::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Ex01opengl *_t = static_cast<Ex01opengl *>(_o);
        switch (_id) {
        case 0: _t->angles((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 1: _t->setShader((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->setPerspective((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->setObject((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->setLighting((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (Ex01opengl::*_t)(QString );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&Ex01opengl::angles)) {
                *result = 0;
            }
        }
    }
}

const QMetaObject Ex01opengl::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_Ex01opengl.data,
      qt_meta_data_Ex01opengl,  qt_static_metacall, 0, 0}
};


const QMetaObject *Ex01opengl::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Ex01opengl::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Ex01opengl.stringdata))
        return static_cast<void*>(const_cast< Ex01opengl*>(this));
    return QGLWidget::qt_metacast(_clname);
}

int Ex01opengl::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void Ex01opengl::angles(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
