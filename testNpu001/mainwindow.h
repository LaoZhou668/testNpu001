#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "rknn_api.h"
#include "postprocess.h"
//#include <sys/time.h>
#include <QThread>
// 文件操作
//#include <qsettings.h>
#include <qfile.h>
#include"rknn_api.h"
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    //screen wid hei
    int fullScrWid,fullScrHei;
    // handle of video
    VideoCapture mVidCap;
    // square Rect of video Center
    Rect sRect;
    // process timer
    int timerProc;
    // 主定时器(Proc)
    void timerEvent(QTimerEvent *event);
    int isVidPlay;

    //jiancekuang
    detect_result_group_t detect_result_group;

    unsigned char* load_modelData(const char *filename, int *model_size);
    int init_model(void);
    // RKNN对象
    float nms_threshold;
    float box_conf_threshold;
    rknn_context ctx;
    const char *post_process_type;
    // Num of RKNN
    int inputNum, outputNum;
    rknn_input inputs[1];
    int width;
    int height;
    // RKNN的输出对象参数
    rknn_input_output_num io_num;
    // output Num Max = 5
    rknn_output outputs[100];
    rknn_tensor_attr output_attrs[200];
    //Esc exit
    void keyPressEvent(QKeyEvent *event);

private:
    Ui::MainWindow *ui;
    // OpenCV Mat ---> QImage(RGB+Gray)
    QImage Mat2QImage(Mat src);
};

#endif // MAINWINDOW_H
