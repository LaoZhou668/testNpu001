#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<qelapsedtimer.h>
#include <qdatetime.h>
#include <QKeyEvent>
#include<QDebug>



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
   ui->setupUi(this);

   // 1 -- 设置视频为全屏模式
   // 首先设置窗口为全屏（标题栏菜单栏已经预先去掉）
   setWindowState(Qt::WindowFullScreen);
   //setWindowFlag(Qt::FramelessWindowHint);
   // 视频窗口预设全屏分辨率
   ui->LBL_IM_SRC->setFixedWidth(1920);
   ui->LBL_IM_SRC->setFixedHeight(1080);
   // 视频窗口全屏并居中
   ui->LBL_IM_SRC->showFullScreen();
   ui->LBL_IM_SRC->setAlignment(Qt::AlignCenter);
   ui->LBL_IM_SRC->setStyleSheet("background-color: rgb(40,44,52)");
   // 视频窗口响应按键消息
   ui->LBL_IM_SRC->installEventFilter(this);

    // 1 -- load Mp4 video
   if(mVidCap.open("/opt/personCar/1080.mp4") == false)
   {
       qDebug("false");
       return;
   }
    //1(mipi)---load camera
    // 打开摄像机
//   if (mVidCap.open(9, cv::CAP_V4L2) == false){
//       qDebug("false");
//       return;
//   }
//    mVidCap = VideoCapture("v4l2src device=/dev/video0 ! video/x-raw,format=NV12,width=1920,height=1080, "
//                           "framerate=30/1 ! videoconvert ! appsink",cv::CAP_GSTREAMER);
//    while(mVidCap.isOpened() == false){
//        //QThread::sleep(100);
//        qDebug("false");
//    }
   // 设置视频格式+分辨率
   mVidCap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
   mVidCap.set(CAP_PROP_FRAME_WIDTH,  1920);
   mVidCap.set(CAP_PROP_FRAME_HEIGHT, 1080);
   // 读取测试帧8f
   Mat tmpSrc;
   for (int i = 0; i < 8; i ++){
       QThread::msleep(33);
       mVidCap.read(tmpSrc);
   }
   qDebug("VIS Camera OK!!!");
    // 裁剪摄像头或者视频分辨率为1080*1080
   sRect.x = (1920-1080)/2;
   sRect.y = 0;
   sRect.width = 1080;
   sRect.height = 1080;

   // 2 -- init RKNN 初始化rknn相关的参数
   //NMS（Non-Maximum Suppression）是一种常用的目标检测算法中的后处理技术，用于消除重叠框（bounding box）和降低误检率。通常情况下，NMS阈值的取值范围为0.3到0.7之间
   nms_threshold = 0.2;
   //box_conf_threshold是用于筛选检测框的一个重要参数，它表示检测框的置信度分数的阈值
   box_conf_threshold = 0.3;
   //RKNN context是进行神经网络推理的关键组成部分，它提供了神经网络推理所需的各种信息和资源，帮助我们快速高效地进行神经网络推理操作
   ctx = 0;
   //指定后处理方式为fp，也就是浮点数
   post_process_type = "fp";
   //载入模型
   init_model();


   // 3 -- start process
   //isVidPlay = 0;
   //定时器
   timerProc = startTimer(33);

}
//将图像缩放并加上黑色边框以适应目标大小
void letterbox(cv::Mat rgb,cv::Mat &img_resize,int target_width,int target_height){

    float shape_0=rgb.rows;
    float shape_1=rgb.cols;
    float new_shape_0=target_height;
    float new_shape_1=target_width;
    //计算缩放比例
    float r=std::min(new_shape_0/shape_0,new_shape_1/shape_1);
    //计算缩放后的宽度和高度 四舍五入
    float new_unpad_0=int(round(shape_1*r));
    float new_unpad_1=int(round(shape_0*r));
    //计算添加黑色边框所需的宽度和高度 dw 和 dh
    float dw=new_shape_1-new_unpad_0;
    float dh=new_shape_0-new_unpad_1;
    dw=dw/2;
    dh=dh/2;
    cv::Mat copy_rgb=rgb.clone();
    if(int(shape_0)!=int(new_unpad_0)&&int(shape_1)!=int(new_unpad_1)){
        //缩放后的高度和宽度不等于输入图像的高度和宽度，就将 copy_rgb 设置为缩放后的图像。然后将缩放后的图像复制给 img_resize
        cv::resize(copy_rgb,img_resize,cv::Size(new_unpad_0,new_unpad_1));
        copy_rgb=img_resize;
    }
    //计算边框的上、下、左、右宽度
    int top=int(round(dh-0.1));
    int bottom=int(round(dh+0.1));
    int left=int(round(dw-0.1));
    int right=int(round(dw+0.1));
    //copyMakeBorder 函数将 copy_rgb 复制到 img_resize，同时添加指定大小的黑色边框
    cv::copyMakeBorder(copy_rgb, img_resize,top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

}
// 主定时器(Proc)
void MainWindow::timerEvent(QTimerEvent *event)
{
    // 如果开始Process
    if (event->timerId() == timerProc){
        //目标分辨率
        const int target_width = 640;
        const int target_height = 640;
        //决定图片处理方式是letter_box自适应
        const char *image_process_mode = "letter_box";
        float resize_scale = 0;

        int h_pad=0;
        int w_pad=0;
        //检测类别
        const char *labels[] = {"pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"};


        // 1 -- Get Src Imagec(1920*1080)
        Mat tmpSrc;
        mVidCap.read(tmpSrc);


        // Get Center ROI(1080*1080)
        // for display
        Mat roi;
        tmpSrc(sRect).copyTo(roi);

        // Get RKNN input image(640*640)
        Mat AIInputImage;

        //BGR->RGB
        cv::cvtColor(roi, AIInputImage, cv::COLOR_BGR2RGB);
        //裁剪
        cv::resize(AIInputImage, AIInputImage, Size(640,640));
        cv::Mat img_resize;
        width=AIInputImage.cols;
        height=AIInputImage.rows;
        // Letter box resize
        //计算输入图像的大小，以便在后续的预处理中进行缩放和填充
        //它首先计算了原始图像的宽高比和目标图像的宽高比
        //原始图像的宽高比大于等于目标图像的宽高比,如果原始图像的宽高比大于等于目标图像的宽高比，则需要在图像的高度方向上进行填充，否则在宽度方向上进行填充
        //然后计算了缩放比例，使得原始图像能够缩放到与目标图像相同的尺寸,最后计算出了需要填充的像素数目，以便将图像填充为目标图像的大小
        float img_wh_ratio = (float) width / (float) height;
        float input_wh_ratio = (float) target_width / (float) target_height;
        int resize_width;
        int resize_height;
        if (img_wh_ratio >= input_wh_ratio) {
            //pad height dim
            resize_scale = (float) target_width / (float) width;
            resize_width = target_width;
            resize_height = (int) ((float) height * resize_scale);
            w_pad = 0;
            h_pad = (target_height - resize_height) / 2;
        } else {
            //pad width dim
            resize_scale = (float) target_height / (float) height;
            resize_width = (int) ((float) width * resize_scale);
            resize_height = target_height;
            w_pad = (target_width - resize_width) / 2;;
            h_pad = 0;
        }
        if(strcmp(image_process_mode,"letter_box")==0){
            letterbox(AIInputImage,img_resize,target_width,target_height);
        }else {
            cv::resize(AIInputImage, img_resize, cv::Size(target_width, target_height));
        }





        // 2 -- RKNN Process
        //将图像输入
        inputs[0].buf = AIInputImage.data;
        //
        rknn_inputs_set(ctx, io_num.n_input, inputs);
//         QElapsedTimer timer;

//        timer.start();
        int ret = rknn_run(ctx, NULL);
        if (ret < 0) {
            printf("ctx error ret=%d\n", ret);
        }
//        double time = (double)timer.nsecsElapsed()/(double)1000000;
//        qDebug("%f", time);
        rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);




        // 3 -- RKNN PostProcess
//        detect_result_group_t detect_result_group;
        std::vector<float> out_scales;
        std::vector<uint8_t> out_zps;
        for (int i = 0; i < io_num.n_output; ++i)
        {
            out_scales.push_back(output_attrs[i].scale);
            out_zps.push_back(output_attrs[i].zp);
        }
        if (strcmp(post_process_type, "u8") == 0) {
            post_process_u8((uint8_t *) outputs[0].buf, (uint8_t *) outputs[1].buf, (uint8_t *) outputs[2].buf,
                            height, width,
                            h_pad, w_pad, resize_scale, box_conf_threshold, nms_threshold, out_zps, out_scales,
                            &detect_result_group, labels);
        } else if (strcmp(post_process_type, "fp") == 0) {
            post_process_fp((float *) outputs[0].buf, (float *) outputs[1].buf, (float *) outputs[2].buf, height,
                            width,
                            h_pad, w_pad, resize_scale, box_conf_threshold, nms_threshold, &detect_result_group, labels);
        }
//        post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, 640, 640,
//                    box_conf_threshold, nms_threshold, 1.0f, 1.0f, out_zps, out_scales, &detect_result_group);
        //free outputs
        rknn_outputs_release(ctx, io_num.n_output, outputs);





        detect_result_group_t detect_result_groups = detect_result_group;
        // 3 -- Draw Objects
        char text[256];
        for (int i = 0; i < detect_result_groups.count; i++)
        {
            detect_result_t *det_result = &(detect_result_groups.results[i]);
            sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
            printf("%s @ (%d %d %d %d) %f\n",
                   det_result->name,
                   det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
                   det_result->prop);
            int x1 = det_result->box.left*1.6875;   // 1080/640
            int y1 = det_result->box.top*1.6875;
            int x2 = det_result->box.right*1.6875;
            int y2 = det_result->box.bottom*1.6875;
            rectangle(roi, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 1);
            //putText(roi, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        Mat litIm;
        cv::resize(roi, litIm, Size(1080, 1080));
        QImage Qim = Mat2QImage(litIm);
        ui->LBL_IM_SRC->setPixmap(QPixmap::fromImage(Qim));
        }   // if 如果开始Process



}   // Main Timer


int MainWindow::init_model(void)
{
    // 1 -- Create the neural network
    //模型数据的大小，以字节为单位
    int model_data_size;
    //yolov5s-640-640.rknn best.rknn
    unsigned char *model_data = load_modelData("/opt/model/RK356X/best.rknn", &model_data_size);
    //用于初始化RKNN context
    //调用rknn_init()函数，可以将模型数据加载到RKNN context中，并进行必要的初始化操作，例如分配内存、加载驱动程序等。初始化成功后，可以使用该context来进行神经网络的推理操作
    int ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    //返回值，表示初始化的结果。如果初始化成功，ret的值为0；否则，表示初始化失败
    if (ret < 0){
        qDebug("rknn_init error ret=%d", ret);
        return ret;
    }

    // 2 -- read version
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(rknn_sdk_version));
    if (ret < 0){
        qDebug("rknn_init error ret=%d", ret);
        return ret;
    }
    qDebug("sdk version: %s driver version: %s", version.api_version,
               version.drv_version);

    // 3 -- shuru1+shuchu3
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0){
        qDebug("rknn_init error ret=%d", ret);
        return ret;
    }
    qDebug("model input num: %d, output num: %d", io_num.n_input,
           io_num.n_output);

    // input rknn tensor attribute
    inputNum = io_num.n_input;
    rknn_tensor_attr input_attrs[inputNum];
    //memset(input_attrs, 0, sizeof(rknn_tensor_attr)*(io_num.n_input));
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < inputNum; i++){
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0){
            qDebug("rknn_init error ret=%d", ret);
            return ret;
        }
        //dump_tensor_attr(&(input_attrs[i]));
    }

    // output rknn tensor attribute
    rknn_tensor_attr output_attrs[io_num.n_output];
    outputNum = io_num.n_output;
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < outputNum; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                         sizeof(rknn_tensor_attr));
        //dump_tensor_attr(&(output_attrs[i]));
    }

    int channel = 3;
    width = 0;
    height = 0;
    // shuru attribute is CHW or HWC
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW){
        qDebug("model is NCHW input fmt");
        //channel = input_attrs[0].dims[1];
        width = input_attrs[0].dims[0];
        height = input_attrs[0].dims[1];
    }
    else{
        qDebug("model is NHWC input fmt");
        width = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        //channel = input_attrs[0].dims[3];
    }
    qDebug("model input height=%d, width=%d, channel=%d", height, width,
           channel);

    // DingYi input data
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    //inputs[0].size = 640 * 640 * 3;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    // DingYi input data
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < outputNum; i++)
    {
        outputs[i].want_float = 0;
    }
    //printf("img.cols: %d, img.rows: %d\n", img_resize.cols, img_resize.rows);

    return 0;
}


// Get RKNN Data
unsigned char* MainWindow::load_modelData(const char *filename, int *model_size)
{
    FILE *fp;
    fp = fopen(filename, "rb");
    if (NULL == fp){
        qDebug("Open RKNN failed!!", filename);
        return NULL;
    }
    // Get Model Size
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    // Get Model Data
    fseek(fp, 0, SEEK_SET);
    unsigned char *data = (unsigned char *)malloc(size);
    fread(data, 1, size, fp);
    fclose(fp);
    // return para
    *model_size = size;
    qDebug("%s",data);
    return data;

}

MainWindow::~MainWindow()
{
    delete ui;
}

// OpenCV Mat ---> QImage(RGB+Gray)
QImage MainWindow::Mat2QImage(Mat src)
{
    QImage imag;

    // RGB
    if (src.channels() == 3){
        cvtColor(src, src, CV_BGR2RGB);
        imag = QImage((const unsigned char *)(src.data), src.cols, src.rows,
                      src.cols*src.channels(), QImage::Format_RGB888);
    }
    // Gray
    else if (src.channels() == 1){
        imag = QImage((const unsigned char *)(src.data), src.cols, src.rows,
                      src.cols*src.channels(), QImage::Format_Grayscale8);
    }
    // others
    else{
        imag = QImage((const unsigned char *)(src.data), src.cols, src.rows,
                      src.cols*src.channels(), QImage::Format_RGB888);
    }
    return imag;
}   // Mat2QImage()




void MainWindow::keyPressEvent(QKeyEvent *event){
    if(event->key() == Qt::Key_Escape){
        QApplication::closeAllWindows();
    }
}
