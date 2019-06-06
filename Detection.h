#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_Detection.h"
#include<opencv2/opencv.hpp>
#include<iostream>
//下面三行保证中文不乱码
#if _MSC_VER >= 1600  
#pragma execution_character_set("utf-8")  
#endif

using namespace std;
using namespace cv;

class Detection : public QMainWindow
{
	Q_OBJECT

public:
	explicit Detection(QWidget *parent = Q_NULLPTR);
	~Detection();

	Mat srcImage, thresholdImage, blurImage, contrastStretchImage, medianImage, restoreImage;
	Mat morhpologyImage, getOptimalDFT_Correction_Image, srcImageDFT, thresholdImageboundary;
	Mat morhpology_dilateImage, closeImage, morhpologyImage2, morhpologyImagetem;
	float DFT_Object_Angle = 0;
	Point ImageCenter;
	int parameter;
	int area_threshold_parameter_count = 0;
	int find_vertical_lines_flag;//用于先检测是否有竖线，对缺陷的定位联通区域有影响，顺序不可变
	static int gray_level;

	static double Gap_Rate;
	static double Overlap_Rate;
	static double Missing_Rate;
	static double Twist_Rate;
	static double Cleavage_Rate;
	static double Fold_Rate;
	static double Bridge_Rate;
	static double Concentrated_Rate;
	static double Scratches_Rate;

	QImage srcimg;
	Mat getOptimalDFT_Correction(Mat srcImage);
	Mat contrastStretch(Mat srcImage);
	QImage cvMat2QImage(const cv::Mat& mat);
	int getOstu(const Mat & in);
	int white_point_rate(Mat testImage);
	double cal_mean_stddev(Mat testImage);
	double Detection::compute_grad(Mat testImage);
	void getglcm_horison(Mat& input, Mat& dst);
	void Detection::getglcm_vertical(Mat& input, Mat& dst);
	void getglcm_45(Mat& input, Mat& dst);
	void Detection::getglcm_135(Mat& input, Mat& dst);
	void feature_computer(Mat&src, double& Asm, double& Eng, double& Con, double& Idm);
	void Spss_Fisher_statics(int, void*);

private slots:

	void on_openTestFile_triggered();

	void on_myExit_triggered();

	void on_openCustomeFile_triggered();

	void on_restore_triggered();

	void on_Clear_triggered();

	void on_rectification_triggered();

	void area_threshold_parameter(int, void*);

	void on_findboundary_triggered();

	void on_findVerticalLines_triggered();

	void on_findHorizontalLines_triggered();

	void on_finddetect_triggered();

	void on_ALLcaculate_triggered();

	void on_about_triggered();

	void on_typeaction_triggered();

private:
	Ui::DetectionClass *ui;

};
