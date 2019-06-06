#include "Detection.h"
#include <QMessageBox>
#include <QFileDialog>
#include <QTextCodec>

double Detection::Gap_Rate = 0;
double Detection::Overlap_Rate = 0;
double Detection::Missing_Rate = 0;
double Detection::Twist_Rate = 0;
double Detection::Cleavage_Rate = 0;
double Detection::Fold_Rate = 0;
double Detection::Bridge_Rate = 0;
double Detection::Concentrated_Rate = 0;
double Detection::Scratches_Rate = 0;
int Detection::gray_level = 16;

Detection::Detection(QWidget *parent)
	: QMainWindow(parent)
{
	ui->setupUi(this);
}

Detection::~Detection()
{
	delete ui;
}

//打开测试文件
void Detection::on_openTestFile_triggered()
{
	srcImage = cv::imread("test.bmp");
	if (!srcImage.data)
	{
		QMessageBox msgBox;
		msgBox.setText(tr("The test image is not found, you could try：1)Copy new image under the current directory, rename test.jpg. 2)Use method custom."));
		msgBox.exec();
	}
	else
	{
		cv::cvtColor(srcImage, srcImage, CV_BGR2GRAY);
		srcimg = cvMat2QImage(srcImage);
		//srcimg = QImage((const unsigned char*)(srcImage.data), srcImage.cols, srcImage.rows, srcImage.cols*srcImage.channels(),QImage::Format_RGB888);
		ui->label1_1->clear();
		srcimg = srcimg.scaled(ui->label1_1->width(), ui->label1_1->height());
		ui->label1_1->setPixmap(QPixmap::fromImage(srcimg));
		//ui->processPushButton->setEnabled(true);
		//   ui->label1->resize(ui->label1->pixmap()->size());//设置当前标签为图像大小
		// ui->label1->resize(img.width(),img.height());

		//this->setWidget(label1);
	}
}

//打开自定义文件
void Detection::on_openCustomeFile_triggered()
{
	QString filename = QFileDialog::getOpenFileName(this, tr("Open Image"), "", tr("Image File(*.bmp *.jpg *.jpeg *.png)"));
	QTextCodec *code = QTextCodec::codecForName("gb18030");
	std::string name = code->fromUnicode(filename).data();
	srcImage = cv::imread(name);
	if (!srcImage.data)
	{
		QMessageBox msgBox;
		msgBox.setText(tr("Not Found!"));
		msgBox.exec();
	}
	else
	{
		cv::cvtColor(srcImage, srcImage, CV_BGR2GRAY);
		srcimg = cvMat2QImage(srcImage);
		//srcimg = QImage((const unsigned char*)(srcImage.data), srcImage.cols, srcImage.rows, srcImage.cols*srcImage.channels(), QImage::Format_RGB888);
		ui->label1_1->clear();
		srcimg = srcimg.scaled(ui->label1_1->width(), ui->label1_1->height());
		ui->label1_1->setPixmap(QPixmap::fromImage(srcimg));
		//ui->processPushButton->setEnabled(true);
		//   ui->label1->resize(ui->label1->pixmap()->size());//设置当前标签为图像大小
		// ui->label1->resize(img.width(),img.height());

		//this->setWidget(label1);
	}
}

//复原
void Detection::on_restore_triggered()
{
	// cv::flip(srcImage,dstImage,-1);
	srcImage.copyTo(restoreImage);
	srcimg = cvMat2QImage(restoreImage);
	srcimg = srcimg.scaled(ui->label1_1->size());
	ui->label1_1->setPixmap(QPixmap::fromImage(srcimg));

}

//清除
void Detection::on_Clear_triggered()
{
	//菜单：文件=>清除
	//清除标签1的内容。setText(tr("新窗口")
	ui->label1_1->setText(tr("原始图像"));
	//清除标签2的内容。
	ui->label1_2->setText(tr("校正图像"));
	ui->label1_3->setText(tr("边界图像"));
	ui->label1_4->setText(tr("连接处图像"));
	ui->label1_5->setText(tr("交界处图像"));
	ui->label1_6->setText(tr("缺陷图像"));
	ui->label2_1->setText(tr("parameter："));
	ui->label2_2->setText(tr("Angel："));
	ui->label2_3->setText(tr("boundary:"));
	ui->label2_4->setText(tr("vertical number:"));
	ui->label2_5->setText(tr("horizontal number:"));
	ui->label2_6->setText(tr("detection:"));
	ui->label3_1->clear();
	ui->label3_2->clear();
	ui->label3_3->clear();
	ui->label3_3->clear();
	ui->label3_4->clear();
	ui->label3_5->clear();
	ui->label3_6->clear();
	ui->label4_1->clear();
	ui->label4_2->clear();
	ui->label4_3->setText(tr("Type:"));
	ui->label4_4->clear();
	ui->label->clear();
}

//退出
void Detection::on_myExit_triggered()
{
	exit(0);
}

//Mat类型到Qimage类型的转换
QImage Detection::cvMat2QImage(const cv::Mat& mat)
{
	// 8-bits unsigned, NO. OF CHANNELS = 1  
	if (mat.type() == CV_8UC1)
	{
		QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
		// Set the color table (used to translate colour indexes to qRgb values)  
		image.setColorCount(256);
		for (int i = 0; i < 256; i++)
		{
			image.setColor(i, qRgb(i, i, i));
		}
		// Copy input Mat  
		uchar *pSrc = mat.data;
		for (int row = 0; row < mat.rows; row++)
		{
			uchar *pDest = image.scanLine(row);
			memcpy(pDest, pSrc, mat.cols);
			pSrc += mat.step;
		}
		return image;
	}
	// 8-bits unsigned, NO. OF CHANNELS = 3  
	else if (mat.type() == CV_8UC3)
	{
		// Copy input Mat  
		const uchar *pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat  
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
		return image.rgbSwapped();
	}
}

//前处理、包含在图像校正里面
void Detection::on_rectification_triggered()
{
	srcImageDFT = getOptimalDFT_Correction(srcImage);//最后在原图中显示
	QImage QsrcImageDFT;
	QsrcImageDFT = cvMat2QImage(srcImageDFT);
	ui->label1_2->clear();
	QsrcImageDFT = QsrcImageDFT.scaled(ui->label1_2->width(), ui->label1_2->height());
	ui->label1_2->setPixmap(QPixmap::fromImage(QsrcImageDFT));
	if (abs(DFT_Object_Angle) < 3)
		ui->label3_2->setText(tr("小于3度"));
	else
	{
		QString tempStr;
		ui->label3_2->setText(tempStr.setNum(DFT_Object_Angle));
	}	

	double time0 = static_cast<double>(getTickCount());
	//获取ostu半自适应性阈值
	parameter = getOstu(srcImage) * 0.65;
	QString tempStr_1;
	ui->label3_1->setText(tempStr_1.setNum(parameter));
	//cout << "The return value of getOstu is: " << parameter << endl;
	//cout << endl;

	//最大灰度值拉伸
	contrastStretchImage = contrastStretch(srcImage);

	area_threshold_parameter(0, 0);

	getOptimalDFT_Correction_Image = getOptimalDFT_Correction(contrastStretchImage);
	//namedWindow("getOptimalDFT_Correction_Image", WINDOW_NORMAL);
	//imshow("getOptimalDFT_Correction_Image", getOptimalDFT_Correction_Image);

	//滤波，使图像更平滑
	blur(getOptimalDFT_Correction_Image, blurImage, Size(5, 5));

	//阈值化
	threshold(blurImage, thresholdImage, parameter, 255, THRESH_BINARY_INV);
	//namedWindow("thresholdImageWindow", WINDOW_NORMAL);
	//imshow("thresholdImageWindow", thresholdImage);
}

//图像校正
Mat Detection::getOptimalDFT_Correction(Mat srcImage)
{
#define GRAY_THRESH 150  
#define HOUGH_VOTE 80 //霍夫检测的投票数，即需要多少点才能确定一条直线，根据lines窗口反馈调整  

	Mat srcImg;
	srcImg = srcImage.clone();
	//获取图像中心点坐标  
	Point center(srcImg.cols / 2, srcImg.rows / 2);
	ImageCenter = center;

	//图像延扩  
	//设置四边尺寸，用getOptimalDFTSize()返回最优离散傅立叶变换（DFT）尺寸  
	//以BORDER_CONSTANT方法延扩图像，用白色填充空白部分  
	Mat padded;
	int opWidth = getOptimalDFTSize(srcImg.rows);
	int opHeight = getOptimalDFTSize(srcImg.cols);
	copyMakeBorder(srcImg, padded, 0, opWidth - srcImg.rows, 0, opHeight - srcImg.cols, BORDER_CONSTANT, Scalar::all(0));

	//DFT  
	//DFT要分别计算实部和虚部，把要处理的图像作为输入的实部、一个全零的图像作为输入的虚部  
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat comImg;
	merge(planes, 2, comImg); //实虚部合并  
	dft(comImg, comImg);

	//获得DFT图像  
	//一般都会用幅度图像来表示图像傅里叶的变换结果（傅里叶谱）  
	//幅度的计算公式：magnitude = sqrt(Re(DFT) ^ 2 + Im(DFT) ^ 2)。  
	//由于幅度的变化范围很大，而一般图像亮度范围只有[0, 255]，容易造成一大片漆黑，只有几个点很亮。所以要用log函数把数值的范围缩小  
	split(comImg, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magMat = planes[0];
	magMat += Scalar::all(1);
	log(magMat, magMat);

	//dft()直接获得的结果中，低频部分位于四角，高频部分位于中间  
	//习惯上会把图像做四等份，互相对调，使低频部分位于图像中心，也就是让频域原点位于中心  
	magMat = magMat(Rect(0, 0, magMat.cols & -2, magMat.rows & -2));
	int cx = magMat.cols / 2;
	int cy = magMat.rows / 2;

	Mat q0(magMat, Rect(0, 0, cx, cy));
	Mat q1(magMat, Rect(0, cy, cx, cy));
	Mat q2(magMat, Rect(cx, cy, cx, cy));
	Mat q3(magMat, Rect(cx, 0, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q2.copyTo(q0);
	tmp.copyTo(q2);

	q1.copyTo(tmp);
	q3.copyTo(q1);
	tmp.copyTo(q3);

	//虽然用log()缩小了数据范围，但仍然不能保证数值都落在[0, 255]之内  
	//所以要先用normalize()规范化到[0, 1]内，再用convertTo()把小数映射到[0, 255]内的整数。结果保存在一幅单通道图像内  
	normalize(magMat, magMat, 0, 1, CV_MINMAX);
	Mat magImg(magMat.size(), CV_8UC1);
	magMat.convertTo(magImg, CV_8UC1, 255, 0);
	//namedWindow("magnitude", WINDOW_NORMAL);
	//imshow("magnitude", magImg);
	//imwrite("imageText_mag.jpg",magImg);  

	//Hough变换要求输入图像是二值的，所以要用threshold()把图像二值化  
	threshold(magImg, magImg, GRAY_THRESH, 255, CV_THRESH_BINARY);
	//imshow("mag_binary", magImg);

	//Hough直线检测  
	vector<Vec2f> lines; //float型二维向量集合“lines”  
	float pi180 = (float)CV_PI / 180; //pi180表示角度制1度  
	Mat linImg(magImg.size(), CV_8UC3); //创建一副新图像，用来显示霍夫检测结果  
	HoughLines(magImg, lines, 1, pi180, HOUGH_VOTE, 0, 0); //霍夫直线检测HoughLines(dst, lines, 分辨率, 单位, 投票数, 参数1, 参数2)  

														   //Hough直线检测结果展示  
	int numLines = lines.size();
	for (int l = 0; l < numLines; l++)
	{
		float rho = lines[l][0], theta = lines[l][1]; //获得第一个向量的rho、theta值  
		Point pt1, pt2; //两个点pt1，pt2  
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(linImg, pt1, pt2, Scalar(255, 0, 0), 3, 8, 0);
	}
	//namedWindow("lines", WINDOW_NORMAL);
	//imshow("lines", linImg);


	//找出除了0度，90度之外的第三条线角度  
	float angel = 0;
	float piThresh = (float)CV_PI / 90; //2度  
	float pi2 = CV_PI / 2;  //pi/2弧度  
	for (int l = 0; l < numLines; l++)
	{
		float theta = lines[l][1];
		if (abs(theta) < piThresh || abs(theta - pi2) < piThresh)
			continue;   //排除掉水平/垂直线  
		else
		{
			angel = theta;
			break;
		}
	}

	//计算旋转角度  
	//图片必须是方形才能计算正确  
	if (angel < pi2)
		angel = angel - CV_PI;
	if (angel != pi2)
	{
		float angelT = srcImg.rows*tan(angel) / srcImg.cols;
		angel = atan(angelT);   //获取非正方形图片的正确转角  
	}
	float angelD = angel * 180 / (float)CV_PI;
	cout << "图片角度为： " << angelD << endl;
	DFT_Object_Angle = angelD;

	Mat dstImg;
	if (abs(angelD) < 3)
		dstImg = srcImage.clone();
	else
	{
		//转正图片  
		Mat rotMat = getRotationMatrix2D(center, angelD, 1.0);
		//Point2f center：表示旋转的中心点  double angle：表示旋转的角度 double scale：图像缩放因子
		dstImg = Mat::zeros(srcImg.size(), CV_8UC1);
		cout << "Size: " << srcImg.size() << endl;
		warpAffine(srcImg, dstImg, rotMat, srcImg.size(), 1, 0, Scalar(255, 255, 255));
	}
	return dstImg;
}

//图像灰度值拉伸
Mat Detection::contrastStretch(Mat srcImage)
{
	Mat resultImage = srcImage.clone();//"=";"clone()";"copyTo"三种拷贝方式，前者是浅拷贝，后两者是深拷贝。  
	int nRows = resultImage.rows;
	int nCols = resultImage.cols;
	//判断图像的连续性  
	if (resultImage.isContinuous())
	{
		nCols = nCols * nRows;
		nRows = 1;
	}
	//图像指针操作  
	uchar *pDataMat;
	int pixMax = 0, pixMin = 255;
	//计算图像的最大最小值  
	for (int j = 0; j < nRows; j++)
	{
		pDataMat = resultImage.ptr<uchar>(j);//ptr<>()得到的是一行指针  
		for (int i = 0; i < nCols; i++)
		{
			if (pDataMat[i] > pixMax)
				pixMax = pDataMat[i];
			if (pDataMat[i] < pixMin)
				pixMin = pDataMat[i];
		}
	}
	//对比度拉伸映射  
	for (int j = 0; j < nRows; j++)
	{
		pDataMat = resultImage.ptr<uchar>(j);
		for (int i = 0; i < nCols; i++)
		{
			pDataMat[i] = (pDataMat[i] - pixMin) * 255 / (pixMax - pixMin);
		}
	}
	return resultImage;
}


int Detection::getOstu(const Mat & in)
{
	int rows = in.rows;
	int cols = in.cols;
	long size = rows * cols;

	float histogram[256] = { 0 };
	for (int i = 0; i < rows; ++i)
	{
		//获取第 i行首像素指针   
		const uchar * p = in.ptr<uchar>(i);
		//对第i 行的每个像素(byte)操作   
		for (int j = 0; j < cols; ++j)
		{
			histogram[int(*p++)]++;
		}
	}
	int threshold;
	long sum0 = 0, sum1 = 0; //存储前景的灰度总和及背景灰度总和    
	long cnt0 = 0, cnt1 = 0; //前景的总个数及背景的总个数    
	double w0 = 0, w1 = 0; //前景及背景所占整幅图像的比例    
	double u0 = 0, u1 = 0;  //前景及背景的平均灰度    
	double variance = 0; //最大类间方差    


	double maxVariance = 0;
	for (int i = 1; i < 256; i++) //一次遍历每个像素    
	{
		sum0 = 0;
		sum1 = 0;
		cnt0 = 0;
		cnt1 = 0;
		w0 = 0;
		w1 = 0;
		for (int j = 0; j < i; j++)
		{
			cnt0 += histogram[j];
			sum0 += j * histogram[j];
		}

		u0 = (double)sum0 / cnt0;
		w0 = (double)cnt0 / size;

		for (int j = i; j <= 255; j++)
		{
			cnt1 += histogram[j];
			sum1 += j * histogram[j];
		}

		u1 = (double)sum1 / cnt1;
		w1 = 1 - w0; // (double)cnt1 / size;    

		variance = w0 * w1 *  (u0 - u1) * (u0 - u1);
		if (variance > maxVariance)
		{
			maxVariance = variance;
			threshold = i;
		}
	}

	return threshold;
}

void Detection::area_threshold_parameter(int, void*)
{

	//【2】定义变量
	MatND dstHist;       // 在cv中用CvHistogram *hist = cvCreateHist
	int dims = 1;
	float hranges[] = { 0, 255 };
	const float *ranges[] = { hranges };   // 这里需要为const类型
	int size = 256;
	int channels = 0;

	//【3】计算图像的直方图
	calcHist(&contrastStretchImage, 1, &channels, Mat(), dstHist, dims, &size, ranges);    // cv 中是cvCalcHist
	int scale = 1;

	Mat dstImage(size * scale, size, CV_8U, Scalar(0));
	//【4】获取最大值和最小值
	double minValue = 0;
	double maxValue = 0;
	minMaxLoc(dstHist, &minValue, &maxValue, 0, 0);  //  在cv中用的是cvGetMinMaxHistValue

													 //【5】绘制出直方图
	int hpt = saturate_cast<int>(0.9 * size);
	for (int i = 0; i < 256; i++)
	{
		float binValue = dstHist.at<float>(i);           //   注意hist中是float类型    而在OpenCV1.0版中用cvQueryHistValue_1D
		int realValue = saturate_cast<int>(binValue * hpt / maxValue);
		rectangle(dstImage, Point(i*scale, size - 1), Point((i + 1)*scale - 1, size - realValue), Scalar(255));
	}
	//namedWindow("一维直方图", WINDOW_NORMAL);
	//imshow("一维直方图", dstImage);

	for (int row = 0; row < contrastStretchImage.rows; row++) {
		for (int col = 0; col < contrastStretchImage.cols; col++) {

			if (contrastStretchImage.at<uchar>(row, col) > 120)
				continue;
			else
			{
				area_threshold_parameter_count++;
			}
		}
	}
	//cout << "the area_threshold_parameter_count number is: " << area_threshold_parameter_count << endl;
}

//检测边界
void Detection::on_findboundary_triggered()
{
	RNG g_rng(12345);
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, dst;

	threshold(srcImage, thresholdImageboundary, 254, 255, THRESH_BINARY_INV);

	Mat element = getStructuringElement(MORPH_RECT, Size(60, 60));
	morphologyEx(thresholdImageboundary, thresholdImageboundary, MORPH_CLOSE, element);
	//namedWindow("thresholdWindow", WINDOW_NORMAL);
	//imshow("thresholdWindow", thresholdImage);

	//求 X方向梯度
	Scharr(thresholdImageboundary, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	//imshow("【效果图】 X方向Scharr", abs_grad_x);

	//求Y方向梯度
	Scharr(thresholdImageboundary, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	//imshow("【效果图】Y方向Scharr", abs_grad_y);

	//合并梯度(近似)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);

	int flag = 1;
	for (int row = 0; row < dst.rows; row++) {
		for (int col = 0; col < dst.cols; col++) {

			if (dst.at<uchar>(row, col) == 0)
				continue;
			else
			{
				flag++;
				break;
			}
		}
	}
	//cout << "Boundary yes or not ?" << endl;
	if (flag == 1)
	{
		ui->label3_3->setText(tr("NOT"));
		ui->label1_3->setText(tr("经检测，此图像不存在边界"));
		//cout << "NOT, 此区域不存在边界" << endl;
	}
	else
	{
		//cout << "YES, 此区域存在边界，具体见图示位置" << endl;
		ui->label3_3->setText(tr("YES"));
		QImage QsrcImageboundary;
		QsrcImageboundary = cvMat2QImage(dst);
		ui->label1_3->clear();
		QsrcImageboundary = QsrcImageboundary.scaled(ui->label1_3->width(), ui->label1_3->height());
		ui->label1_3->setPixmap(QPixmap::fromImage(QsrcImageboundary));
		//namedWindow("boundary yes or not", WINDOW_NORMAL);
		//imshow("boundary yes or not", dst);
	}
	return;
}

//检测连接处
void Detection::on_findVerticalLines_triggered()
{
	RNG g_rng(12345);
	Mat find_vertical_lines_medianImage;
	medianBlur(thresholdImage, find_vertical_lines_medianImage, 7);

	Mat find_vertical_lines_openImage;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(1, 20));
	morphologyEx(find_vertical_lines_medianImage, find_vertical_lines_openImage, MORPH_OPEN, kernel);
	//namedWindow("find_vertical_lines_openImageWindow", WINDOW_NORMAL);
	//imshow("find_vertical_lines_openImageWindow", find_vertical_lines_openImage);

	// 矩形内核膨胀
	Mat find_vertical_lines_dilateImage;
	Mat kernel1 = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));
	dilate(find_vertical_lines_openImage, find_vertical_lines_dilateImage, kernel1);
	//namedWindow("find_vertical_lines_dilateImageWindow", WINDOW_NORMAL);
	//imshow("find_vertical_lines_dilateImageWindow", find_vertical_lines_dilateImage);

	// 寻找轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(find_vertical_lines_dilateImage, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	// 多边形逼近轮廓 + 获取矩形和圆形边界框
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	//vector<Point2f>center(contours.size());
	//vector<float>radius(contours.size());

	//【7】遍历所有部分，逼近边界
	for (unsigned int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);//用指定精度逼近多边形曲线 
		boundRect[i] = boundingRect(Mat(contours_poly[i]));//计算点集的最外面（up-right）矩形边界
														   //minEnclosingCircle(contours_poly[i], center[i], radius[i]);//对给定的 2D点集，寻找最小面积的包围圆形 
	}

	// 【8】绘制多边形轮廓 + 包围的矩形框
	Mat drawing = Mat::zeros(find_vertical_lines_dilateImage.size(), CV_8UC3);
	int xx = 0;
	int yy = 0;
	int areaxxyy = 0;
	float xdistance[80];
	float ydistance[80];
	int number = 0;
	for (int unsigned i = 0; i<contours.size(); i++)
	{
		Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));//随机设置颜色

		xx = abs(boundRect[i].br().x - boundRect[i].tl().x);
		yy = abs(boundRect[i].br().y - boundRect[i].tl().y);
		areaxxyy = xx * yy;
		if (areaxxyy> 3000 && areaxxyy < 40000)
		{
			float ratio = yy / xx;
			if (ratio < 5)
				continue;
			else
			{
				xdistance[number] = xx;
				ydistance[number] = yy;
				number++;
				rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);//绘制矩形
																						 //cvRectangle函数参数： 图片， 左上角， 右下角， 颜色， 线条粗细， 线条类型，点类型
																						 //circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);//绘制圆
			}
		}
		else
			continue;
	}
	//cout << "find_vertical_lines result: " << endl << endl;

	if (number > 10 || number == 0)
	{
		find_vertical_lines_flag = -1;
		ui->label3_4->setText("0");
		ui->label1_4->setText("经检测，此图像不存在间隙");
		//cout << "There is no vertical line!" << endl;
	}
	else
	{
		find_vertical_lines_flag = 1;
		QString tempStr_2;
		ui->label3_4->setText(tempStr_2.setNum(number));
		Gap_Rate = Gap_Rate + 100;
		//ui->label4_4->setText(tr("间隙"));
		//cout << "The number of vertical lines is: " << number << endl;
		//cout << "缺陷类型：间隙" << endl << endl;
		QString ALLInformation;
		for (int i = 0; i < number; i++)
		{
			QString vertical_lines_math,xxx,yyy,aaa;
			QString num;
			vertical_lines_math = "          No." + num.setNum(i+1) +"： " + xxx.setNum(xdistance[i] * 0.01) + "  * "
									 + yyy.setNum(ydistance[i] * 0.01) + " = "
									 + aaa.setNum(xdistance[i] * ydistance[i] * 0.0001);
			ALLInformation = ALLInformation.append(vertical_lines_math)+"\n";

		}
		ui->label4_1->setText(ALLInformation);
		// 显示效果图窗口
		//namedWindow("ResultWindow", WINDOW_NORMAL);
		//imshow("ResultWindow", drawing);

		//【9】Hough直线检测
		Mat find_vertical_lines_houghImage;
		vector<Vec4i> lines;
		HoughLinesP(find_vertical_lines_dilateImage, lines, 1, CV_PI / 180.0, 50, 20.0, 0);
		srcImageDFT.copyTo(find_vertical_lines_houghImage);
		cvtColor(find_vertical_lines_houghImage, find_vertical_lines_houghImage, COLOR_GRAY2BGR);
		for (size_t t = 0; t < lines.size(); t++) {
			Vec4i ln = lines[t];
			line(find_vertical_lines_houghImage, Point(ln[0], ln[1]), Point(ln[2], ln[3]), Scalar(0, 0, 255), 2, 8, 0);
		}
		QImage Qfind_vertical_lines_houghImage = cvMat2QImage(find_vertical_lines_houghImage);
		//srcimg = QImage((const unsigned char*)(srcImage.data), srcImage.cols, srcImage.rows, srcImage.cols*srcImage.channels(), QImage::Format_RGB888);
		ui->label1_4->clear();
		Qfind_vertical_lines_houghImage = Qfind_vertical_lines_houghImage.scaled(ui->label1_4->width(), ui->label1_4->height());
		ui->label1_4->setPixmap(QPixmap::fromImage(Qfind_vertical_lines_houghImage));
		//namedWindow(WINDOWNAMEFINAL, WINDOW_NORMAL);
		//imshow(WINDOWNAMEFINAL, find_vertical_lines_houghImage);
	}
	cout << endl;
}

//检测交界处
void Detection::on_findHorizontalLines_triggered()
{
	RNG g_rng(12345);
	Mat ROI_lines;
	ROI_lines = Mat::zeros(srcImage.size(), CV_8UC1);
	//cvtColor(ROI_lines, ROI_lines, COLOR_GRAY2BGR);
	vector<Vec4i> lines;
	HoughLinesP(thresholdImage, lines, 1, CV_PI / 180.0, 100, 30.0, 0);

	for (size_t t = 0; t < lines.size(); t++)
	{
		Vec4i ln = lines[t];
		line(ROI_lines, Point(ln[0], ln[1]), Point(ln[2], ln[3]), 255, 8, CV_AA, 0);
	}
	//cout << "Findlines result(All lines) as Window: 效果图" << endl << endl;
	int rate_1 = white_point_rate(ROI_lines);
	//cout << "前者：" << rate_1 << endl;

	//namedWindow("WINDOWNAMEFINAL", WINDOW_NORMAL);
	//imshow("WINDOWNAMEFINAL", ROI_lines);

	int MaxcountNumber = 0;
	int sumrow;
	for (int row = 0; row < ROI_lines.rows; row++)
	{
		sumrow = 0;
		for (int col = 0; col < ROI_lines.cols; col++)
		{
			if (ROI_lines.at<uchar>(row, col) != 0)
				sumrow++;
			else
				continue;
		}
		//cout << sumrow << endl;
		if (sumrow > MaxcountNumber)
			MaxcountNumber = sumrow;
	}
	//cout << "横向领域最大计数：" << MaxcountNumber << endl;
	float countNumber = MaxcountNumber * 0.5;
	//cout << countNumber << endl;

	//找边界 对方向有很高的要求，尽量水平
	Mat find_horizontal_lines_Image;
	ROI_lines.copyTo(find_horizontal_lines_Image);
	for (int row = 0; row < ROI_lines.rows; row++) {
		int sumrow = 0;
		for (int col = 0; col < ROI_lines.cols; col++) {

			if (ROI_lines.at<uchar>(row, col) != 0)
				sumrow++;
			else
				continue;
		}

		if (sumrow > countNumber)
		{
			for (int col = 0; col < find_horizontal_lines_Image.cols; col++)
			{
				find_horizontal_lines_Image.at<uchar>(row, col) = 255;
			}
		}
		else
		{
			for (int col = 0; col < find_horizontal_lines_Image.cols; col++)
			{
				find_horizontal_lines_Image.at<uchar>(row, col) = 0;
			}
		}
	}
	int rate_2 = white_point_rate(find_horizontal_lines_Image);
	//cout << "后者：" << rate_2 << endl;
	float whiteRate = (float)rate_2 / (float)rate_1;
	//cout << "比例：" << whiteRate << endl;
	//if (whiteRate > 1.09)
	//	Cleavage_Rate = Cleavage_Rate + 30;
	//cout << "find_horizontal_lines result: " << endl << endl;
	

	//namedWindow("find_horizontal_lines_ImageWindow", WINDOW_NORMAL);
	//imshow("find_horizontal_lines_ImageWindow", find_horizontal_lines_Image);
	QImage Qfind_horizontal_lines_Image = cvMat2QImage(find_horizontal_lines_Image);
	Qfind_horizontal_lines_Image = Qfind_horizontal_lines_Image.scaled(ui->label1_5->width(), ui->label1_5->height());
	ui->label1_5->setPixmap(QPixmap::fromImage(Qfind_horizontal_lines_Image));

	//寻找轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(find_horizontal_lines_Image, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	//多边形逼近轮廓 + 获取矩形和圆形边界框
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	//遍历所有部分，逼近边界
	for (unsigned int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);//用指定精度逼近多边形曲线 
		boundRect[i] = boundingRect(Mat(contours_poly[i]));//计算点集的最外面（up-right）矩形边界
	}

	// 绘制多边形轮廓 + 包围的矩形框
	Mat drawing = Mat::zeros(find_horizontal_lines_Image.size(), CV_8UC3);
	float xdistance = 0;
	float ydistance = 0;
	float area = 0.0;
	int number = 0;

	QString ALLInformation;
	for (int unsigned i = 0; i<contours.size(); i++)
	{
		Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));//随机设置颜色

		xdistance = abs(boundRect[i].br().x - boundRect[i].tl().x);
		ydistance = abs(boundRect[i].br().y - boundRect[i].tl().y);
		if (xdistance * ydistance > 3000)
		{
			number++;
			QString horizontal_lines_math, xxx, yyy, aaa, num;
			horizontal_lines_math = "          No." + num.setNum(number) + "： " + xxx.setNum(xdistance * 0.01) + "  * "
												+ yyy.setNum(ydistance * 0.01) + " = "
												+ aaa.setNum(xdistance * ydistance * 0.0001);
			//cout << "该图示区域经测量，实际长度为： " << xdistance * 0.01 << " mm；" << endl;
			//cout << "该图示区域经测量，实际宽度为： " << ydistance * 0.01 << " mm；" << endl;
			//cout << "该图示区域经测量，实际面积为： " << xdistance * ydistance * 0.0001 << " mm*mm；" << endl;
			if (ydistance > 40)
				horizontal_lines_math = horizontal_lines_math.append("\n 此条直线收到周边缺陷影响较大，参考意义很小，如有需要，建议人工观察预浸料！");
				//cout << "此条直线收到周边缺陷影响较大，参考意义很小，如有需要，建议人工观察预浸料！" << endl;
			ALLInformation = ALLInformation.append(horizontal_lines_math) + "\n";
			rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);//绘制矩形
			// cvRectangle函数参数： 图片， 左上角， 右下角， 颜色， 线条粗细， 线条类型，点类型
		}
		else
			continue;
	}
	//cout << "The number of horizontal lines is: " << number << endl << endl;
	QString tempStr_3;
	ui->label3_5->setText(tempStr_3.setNum(number));
	if (number > 5)
	{
		ALLInformation = ALLInformation.append("此处存在直线状缺陷，与预浸料交界处相似") + "\n";
		Cleavage_Rate = Cleavage_Rate + 30;
		Twist_Rate = Twist_Rate + 30;
	}
	ui->label4_2->setText(ALLInformation);

		//cout << "此处存在直线状缺陷，与预浸料交界处相似" << endl;
	//cout << "find_horizontal_lines result as Window <find_horizontal_lines_ImageWindow>" << endl;
}

//检测面状缺陷
void Detection::on_finddetect_triggered()
{
	RNG g_rng(12345);
	medianBlur(thresholdImage, medianImage, 7);
	//namedWindow("image4Window", WINDOW_NORMAL);
	//imshow("image4Window", medianImage);

	if (find_vertical_lines_flag < 0)
	{
		Mat element = getStructuringElement(MORPH_RECT, Size(20, 30));
		morphologyEx(medianImage, morhpologyImage, MORPH_DILATE, element);
		//namedWindow("morhpologyImageWindow", WINDOW_NORMAL);
		//imshow("morhpologyImageWindow", morhpologyImage);
	}
	else
	{
		Mat openImage;
		Mat element1 = getStructuringElement(MORPH_RECT, Size(15, 1));
		morphologyEx(medianImage, openImage, MORPH_OPEN, element1);
		//namedWindow("openImageWindow", WINDOW_NORMAL);
		//imshow("openImageWindow", openImage);

		Mat element = getStructuringElement(MORPH_RECT, Size(20, 30));
		morphologyEx(openImage, morhpologyImage, MORPH_DILATE, element);
		//namedWindow("morhpologyImageWindow", WINDOW_NORMAL);
		//imshow("morhpologyImageWindow", morhpologyImage);
	}

	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(morhpologyImage, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());

	Mat resultImage = Mat::zeros(morhpologyImage.size(), CV_8UC1);
	Point cc;
	int detect_number = 0;
	int contour_area[80];
	for (size_t t = 0; t < contours.size(); t++) {
		// 面积过滤
		double area = contourArea(contours[t]);
		if (area < area_threshold_parameter_count)
			continue;
		else
		{
			Rect rect = boundingRect(contours[t]);
			float ratio = float(rect.width) / float(rect.height);

			if (ratio < 0.05 || ratio > 20)
				continue;
			else
			{
				if ((area < 110000) && (ratio > 0.125 && ratio < 8))
					Concentrated_Rate = Concentrated_Rate + 50;
				detect_number++;
				contour_area[detect_number] = area;
				drawContours(resultImage, contours, t, 255, -1, 8, Mat(), 0, Point());
				//printf("circle area : %f\n", area);
				Rect rect = boundingRect(contours[t]);
				//printf("circle length : %f\n", arcLength(contours[t], true));
			}
		}
	}
	cout << "Find detects results:" << endl;
	if (detect_number == 0)
	{
		//cout << "There is no detect!" << endl;
		ui->label1_6->setText("经检测，此图像没有面状缺陷");
		ui->label3_6->setText("0");
		//Twist_Rate = Twist_Rate + 15;
	}
	else
	{
		//cout << "The number of detects is " << detect_number << endl;
		QString tempStr_4;
		ui->label3_6->setText(tempStr_4.setNum(detect_number));
		for (int i = 1; i < detect_number + 1; i++)
		{
			float area_i = (float)contour_area[i] / 10000;
			cout << "No. " << i << " detect area is(not hull area): " << area_i << " mm*mm " << endl;
		}
		cout << endl;
		//namedWindow("connectImageWindow", WINDOW_NORMAL);
		//imshow("connectImageWindow", resultImage);

		vector<vector<Point>> contours1;
		vector<Vec4i>hierarchy;
		findContours(resultImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<vector<Point>>hull(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			convexHull(Mat(contours[i]), hull[i], false);
		}

		//绘制轮廓和凸包
		Mat finddetect_hullImage = srcImageDFT.clone();
		for (int i = 0; i < contours.size(); i++)
		{
			//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			//drawContours(finddetect_hullImage, contours, i, 0, 8, 8, vector<Vec4i>(), 0, Point());
			drawContours(finddetect_hullImage, hull, i, 0, 8, 8, vector<Vec4i>(), 0, Point());
		}

		//namedWindow("finddetect_hullImageWindow", WINDOW_NORMAL);
		//imshow("finddetect_hullImageWindow", finddetect_hullImage);
		QImage Qfinddetect_hullImage = cvMat2QImage(finddetect_hullImage);
		Qfinddetect_hullImage = Qfinddetect_hullImage.scaled(ui->label1_6->width(), ui->label1_6->height());
		ui->label1_6->setPixmap(QPixmap::fromImage(Qfinddetect_hullImage));
	}
}

//一步到位计算
void Detection::on_ALLcaculate_triggered()
{
	double time0 = static_cast<double>(getTickCount());
	
	//Step1
	on_rectification_triggered();

	//Step2
	on_findboundary_triggered();

	//Step3
	on_findVerticalLines_triggered();

	//Step4
	on_findHorizontalLines_triggered();

	//Step5
	on_finddetect_triggered();

	//Step6
	on_typeaction_triggered();
	
	time0 = ((double)getTickCount() - time0) / getTickFrequency();
	QString tempStr_time;
	//tempStr_time = "Total time:" + tempStr_time.setNum(time0);
	tempStr_time = tempStr_time.setNum(time0) + " s" ;
	ui->label->setText(tempStr_time);

	//重置缺陷比例值

}

//aboutme
void Detection::on_about_triggered()
{
	
		QMessageBox msgBox;
		QString Aboutme;
		Aboutme = Aboutme.append("程序描述：AFP Automated Inspection System") + "\n";
		Aboutme = Aboutme.append("开发测试所用操作系统： Windows 10 64bit") + "\n";
		Aboutme = Aboutme.append("开发测试所用IDE版本：Visual Studio Enterprise 2017") + "\n";
		Aboutme = Aboutme.append("开发测试所用OpenCV版本：	3.4.1") + "\n";
		Aboutme = Aboutme.append("开发测试所用OpenCV_Contrib版本  3.4.1") + "\n";
		Aboutme = Aboutme.append("开发测试所用QT版本  5.9") + "\n";
		Aboutme = Aboutme.append("2018年5月13日 Created by  @QiaoLei") + "\n";
		Aboutme = Aboutme.append("2018年5月29日 Revised by  @QiaoLei") + "\n";
		Aboutme = Aboutme.append("功能简介：") + "\n";
		Aboutme = Aboutme.append("	1.检测区域的位置，并计算面积大小") + "\n";
		Aboutme = Aboutme.append("	2.检测预浸料边界分界线，计算边界的宽度") + "\n";
		Aboutme = Aboutme.append("	3.检测是否含有碳纤维边界") + "\n";
		Aboutme = Aboutme.append("	4.检测是否具有预浸料连接处，如有，计算长宽，如无，显示无") + "\n";
		msgBox.setText(Aboutme);
		msgBox.exec();
		
}

//计算二值图像中的白点比例
int Detection::white_point_rate(Mat testImage)
{
	int sumwhite = 0;
	for (int row = 0; row < testImage.rows; row++) {
		for (int col = 0; col < testImage.cols; col++) {

			if (testImage.at<uchar>(row, col) != 0)
				sumwhite++;
			else
				continue;
		}
	}
	return sumwhite;
}

//统计缺陷的比例
void Detection::on_typeaction_triggered()
{
	//函数构造部分
	Spss_Fisher_statics(0,0);

	if (Gap_Rate > 100)
		Gap_Rate = 100;
	if (Overlap_Rate > 100)
		Overlap_Rate = 100;
	if (Missing_Rate > 100)
		Missing_Rate = 100;
	if (Twist_Rate > 100)
		Twist_Rate = 100;
	if (Cleavage_Rate > 100)
		Cleavage_Rate = 100;
	if (Fold_Rate > 100)
		Fold_Rate = 100;
	if (Bridge_Rate > 100)
		Bridge_Rate = 100;
	if (Concentrated_Rate > 100)
		Concentrated_Rate = 100;
	if (Scratches_Rate > 100)
		Scratches_Rate = 100;

	QString * ph = new QString;
	*ph = (*ph).setNum(Scratches_Rate).append("%");
	ui->label6_1->setText(*ph);
	delete ph;

	QString * pn = new QString;
	*pn = (*pn).setNum(Gap_Rate).append("%");
	ui->label6_2->setText(*pn);
	delete pn;
	
	QString * pa = new QString;
	*pa = (*pa).setNum(Overlap_Rate).append("%");
	ui->label6_3->setText(*pa);
	delete pa;

	QString * pb = new QString;
	*pb = (*pb).setNum(Missing_Rate).append("%");
	ui->label6_4->setText(*pb);
	delete pb;

	QString * pc = new QString;
	*pc = (*pc).setNum(Twist_Rate).append("%");
	ui->label6_5->setText(*pc);
	delete pc;

	QString * pd = new QString;
	*pd = (*pd).setNum(Cleavage_Rate).append("%");
	ui->label6_6->setText(*pd);
	delete pd;

	QString * pe = new QString;
	*pe = (*pe).setNum(Fold_Rate).append("%");
	ui->label6_7->setText(*pe);
	delete pe;

	QString * pf = new QString;
	*pf = (*pf).setNum(Bridge_Rate).append("%");
	ui->label6_8->setText(*pf);
	delete pf;

	QString * pg = new QString;
	*pg = (*pg).setNum(Concentrated_Rate).append("%");
	ui->label6_9->setText(*pg);
	delete pg;

	QString * p_rate = new QString;
	if (Scratches_Rate>70)
		*p_rate = (*p_rate).append(tr(" 划痕"));
	if (Gap_Rate > 70)
		*p_rate = (*p_rate).append(tr(" 间隙"));
	if (Overlap_Rate>70)
		*p_rate = (*p_rate).append(tr(" 重叠"));
	if (Missing_Rate>70)
		*p_rate = (*p_rate).append(tr(" 缺失"));
	if (Twist_Rate>70)
		*p_rate = (*p_rate).append(tr(" 扭转"));
	if (Cleavage_Rate>70)
		*p_rate = (*p_rate).append(tr(" 劈裂"));
	if (Fold_Rate>70)
		*p_rate = (*p_rate).append(tr(" 褶皱"));
	if (Bridge_Rate>70)
		*p_rate = (*p_rate).append(tr(" 架桥"));
	if (Concentrated_Rate>70)
		*p_rate = (*p_rate).append(tr(" 局部"));
	*p_rate = (*p_rate).trimmed();
	ui->label4_4->setText(*p_rate);
	delete p_rate;

	Gap_Rate = 0;
	Overlap_Rate = 0;
	Missing_Rate = 0;
	Twist_Rate = 0;
	Cleavage_Rate = 0;
	Fold_Rate = 0;
	Bridge_Rate = 0;
	Concentrated_Rate = 0;
}

//计算标准差
double Detection::cal_mean_stddev(Mat testImage) {
	//void mean_stddev_AvG(Mat testImage)
	Mat src = testImage.clone();
	Mat mat_mean, mat_stddev;
	meanStdDev(src, mat_mean, mat_stddev);
	double m, s;
	m = mat_mean.at<double>(0, 0);
	s = mat_stddev.at<double>(0, 0);
	//cout << "灰度均值是：" << m << endl;
	//cout << "标准差是：" << s << endl;
	return s;
}

//计算梯度
double Detection::compute_grad(Mat testImage)
{
	Mat img = testImage.clone();
	img.convertTo(img, CV_64FC1);
	double tmp = 0;
	int rows = img.rows - 1;
	int cols = img.cols - 1;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double dx = img.at<double>(i, j + 1) - img.at<double>(i, j);
			double dy = img.at<double>(i + 1, j) - img.at<double>(i, j);
			double ds = std::sqrt((dx*dx + dy * dy) / 2);
			tmp += ds;
		}
	}
	double imageAvG = tmp / (rows*cols);
	//cout << "平均梯度是：" << imageAvG << endl;
	return imageAvG;
}

//0度灰度共生矩阵
void Detection::getglcm_horison(Mat& input, Mat& dst)
{
	Mat src = input;
	CV_Assert(1 == src.channels());
	src.convertTo(src, CV_32S);
	int height = src.rows;
	int width = src.cols;
	int max_gray_level = 0;
	for (int j = 0; j < height; j++)//寻找像素灰度最大值
	{
		int* srcdata = src.ptr<int>(j);
		for (int i = 0; i < width; i++)
		{
			if (srcdata[i] > max_gray_level)
			{
				max_gray_level = srcdata[i];
			}
		}
	}
	max_gray_level++;//像素灰度最大值加1即为该矩阵所拥有的灰度级数
	if (max_gray_level > 16)//若灰度级数大于16，则将图像的灰度级缩小至16级，减小灰度共生矩阵的大小。
	{
		for (int i = 0; i < height; i++)
		{
			int*srcdata = src.ptr<int>(i);
			for (int j = 0; j < width; j++)
			{
				srcdata[j] = (int)srcdata[j] / gray_level;
			}
		}

		dst.create(gray_level, gray_level, CV_32SC1);
		dst = Scalar::all(0);
		for (int i = 0; i < height; i++)
		{
			int*srcdata = src.ptr<int>(i);
			for (int j = 0; j < width - 1; j++)
			{
				int rows = srcdata[j];
				int cols = srcdata[j + 1];
				dst.ptr<int>(rows)[cols]++;
			}
		}
	}
	else//若灰度级数小于16，则生成相应的灰度共生矩阵
	{
		dst.create(max_gray_level, max_gray_level, CV_32SC1);
		dst = Scalar::all(0);
		for (int i = 0; i < height; i++)
		{
			int*srcdata = src.ptr<int>(i);
			for (int j = 0; j < width - 1; j++)
			{
				int rows = srcdata[j];
				int cols = srcdata[j + 1];
				dst.ptr<int>(rows)[cols]++;
			}
		}
	}
}

//90度灰度共生矩阵
void Detection::getglcm_vertical(Mat& input, Mat& dst)
{
	Mat src = input;
	CV_Assert(1 == src.channels());
	src.convertTo(src, CV_32S);
	int height = src.rows;
	int width = src.cols;
	int max_gray_level = 0;
	for (int j = 0; j < height; j++)
	{
		int* srcdata = src.ptr<int>(j);
		for (int i = 0; i < width; i++)
		{
			if (srcdata[i] > max_gray_level)
			{
				max_gray_level = srcdata[i];
			}
		}
	}
	max_gray_level++;
	if (max_gray_level > 16)
	{
		for (int i = 0; i < height; i++)//将图像的灰度级缩小至16级，减小灰度共生矩阵的大小。
		{
			int*srcdata = src.ptr<int>(i);
			for (int j = 0; j < width; j++)
			{
				srcdata[j] = (int)srcdata[j] / gray_level;
			}
		}

		dst.create(gray_level, gray_level, CV_32SC1);
		dst = Scalar::all(0);
		for (int i = 0; i < height - 1; i++)
		{
			int*srcdata = src.ptr<int>(i);
			int*srcdata1 = src.ptr<int>(i + 1);
			for (int j = 0; j < width; j++)
			{
				int rows = srcdata[j];
				int cols = srcdata1[j];
				dst.ptr<int>(rows)[cols]++;
			}
		}
	}
	else
	{
		dst.create(max_gray_level, max_gray_level, CV_32SC1);
		dst = Scalar::all(0);
		for (int i = 0; i < height - 1; i++)
		{
			int*srcdata = src.ptr<int>(i);
			int*srcdata1 = src.ptr<int>(i + 1);
			for (int j = 0; j < width; j++)
			{
				int rows = srcdata[j];
				int cols = srcdata1[j];
				dst.ptr<int>(rows)[cols]++;
			}
		}
	}
}

//45度灰度共生矩阵
void Detection::getglcm_45(Mat& input, Mat& dst)
{
	Mat src = input;
	CV_Assert(1 == src.channels());
	src.convertTo(src, CV_32S);
	int height = src.rows;
	int width = src.cols;
	int max_gray_level = 0;
	for (int j = 0; j < height; j++)
	{
		int* srcdata = src.ptr<int>(j);
		for (int i = 0; i < width; i++)
		{
			if (srcdata[i] > max_gray_level)
			{
				max_gray_level = srcdata[i];
			}
		}
	}
	max_gray_level++;
	if (max_gray_level > 16)
	{
		for (int i = 0; i < height; i++)//将图像的灰度级缩小至16级，减小灰度共生矩阵的大小。
		{
			int*srcdata = src.ptr<int>(i);
			for (int j = 0; j < width; j++)
			{
				srcdata[j] = (int)srcdata[j] / gray_level;
			}
		}

		dst.create(gray_level, gray_level, CV_32SC1);
		dst = Scalar::all(0);
		for (int i = 0; i < height - 1; i++)
		{
			int*srcdata = src.ptr<int>(i);
			int*srcdata1 = src.ptr<int>(i + 1);
			for (int j = 0; j < width - 1; j++)
			{
				int rows = srcdata[j];
				int cols = srcdata1[j + 1];
				dst.ptr<int>(rows)[cols]++;
			}
		}
	}
	else
	{
		dst.create(max_gray_level, max_gray_level, CV_32SC1);
		dst = Scalar::all(0);
		for (int i = 0; i < height - 1; i++)
		{
			int*srcdata = src.ptr<int>(i);
			int*srcdata1 = src.ptr<int>(i + 1);
			for (int j = 0; j < width - 1; j++)
			{
				int rows = srcdata[j];
				int cols = srcdata1[j + 1];
				dst.ptr<int>(rows)[cols]++;
			}
		}
	}
}

//135度灰度共生矩阵
void Detection::getglcm_135(Mat& input, Mat& dst)
{
	Mat src = input;
	CV_Assert(1 == src.channels());
	src.convertTo(src, CV_32S);
	int height = src.rows;
	int width = src.cols;
	int max_gray_level = 0;
	for (int j = 0; j < height; j++)
	{
		int* srcdata = src.ptr<int>(j);
		for (int i = 0; i < width; i++)
		{
			if (srcdata[i] > max_gray_level)
			{
				max_gray_level = srcdata[i];
			}
		}
	}
	max_gray_level++;
	if (max_gray_level > 16)
	{
		for (int i = 0; i < height; i++)//将图像的灰度级缩小至16级，减小灰度共生矩阵的大小。
		{
			int*srcdata = src.ptr<int>(i);
			for (int j = 0; j < width; j++)
			{
				srcdata[j] = (int)srcdata[j] / gray_level;
			}
		}

		dst.create(gray_level, gray_level, CV_32SC1);
		dst = Scalar::all(0);
		for (int i = 0; i < height - 1; i++)
		{
			int*srcdata = src.ptr<int>(i);
			int*srcdata1 = src.ptr<int>(i + 1);
			for (int j = 1; j < width; j++)
			{
				int rows = srcdata[j];
				int cols = srcdata1[j - 1];
				dst.ptr<int>(rows)[cols]++;
			}
		}
	}
	else
	{
		dst.create(max_gray_level, max_gray_level, CV_32SC1);
		dst = Scalar::all(0);
		for (int i = 0; i < height - 1; i++)
		{
			int*srcdata = src.ptr<int>(i);
			int*srcdata1 = src.ptr<int>(i + 1);
			for (int j = 1; j < width; j++)
			{
				int rows = srcdata[j];
				int cols = srcdata1[j - 1];
				dst.ptr<int>(rows)[cols]++;
			}
		}
	}
}

//计算特征值
void Detection::feature_computer(Mat&src, double& Asm, double& Eng, double& Con, double& Idm)
{
	int height = src.rows;
	int width = src.cols;
	int total = 0;
	for (int i = 0; i < height; i++)
	{
		int*srcdata = src.ptr<int>(i);
		for (int j = 0; j < width; j++)
		{
			total += srcdata[j];//求图像所有像素的灰度值的和
		}
	}

	Mat copy;
	copy.create(height, width, CV_64FC1);
	for (int i = 0; i < height; i++)
	{
		int*srcdata = src.ptr<int>(i);
		double*copydata = copy.ptr<double>(i);
		for (int j = 0; j < width; j++)
		{
			copydata[j] = (double)srcdata[j] / (double)total;//图像每一个像素的的值除以像素总和
		}
	}


	for (int i = 0; i < height; i++)
	{
		double*srcdata = copy.ptr<double>(i);
		for (int j = 0; j < width; j++)
		{
			Asm += srcdata[j] * srcdata[j];//能量
			if (srcdata[j]>0)
				Eng -= srcdata[j] * log(srcdata[j]);//熵             
			Con += (double)(i - j)*(double)(i - j)*srcdata[j];//对比度
			Idm += srcdata[j] / (1 + (double)(i - j)*(double)(i - j));//逆差矩
		}
	}
}

//缺陷类型检测
void Detection::Spss_Fisher_statics(int, void*)
{
	//判别所用的参数
	double discriminant_parameter_1;//占30分
	double discriminant_parameter_2;//占20分
	double discriminant_parameter_3;//占15分
	double discriminant_parameter_4;//占15分
	double discriminant_parameter_5;//占10分
	double discriminant_parameter_6;//占5分
	double discriminant_parameter_7;//占5分

	//Caculation1
	double Image_deviation = cal_mean_stddev(contrastStretchImage);//参数1
	
	//Caculation2
	double Image_grad = compute_grad(contrastStretchImage);//参数2

	Mat dst_horison, dst_vertical, dst_45, dst_135;
	getglcm_horison(contrastStretchImage, dst_horison);
	getglcm_vertical(contrastStretchImage, dst_vertical);
	getglcm_45(contrastStretchImage, dst_45);
	//getglcm_135(contrastStretchImage, dst_135);

	
	//Caculation3
	double eng_horison = 0, con_horison = 0, idm_horison = 0, asm_horison = 0;
	feature_computer(dst_horison, asm_horison, eng_horison, con_horison, idm_horison);
	//--------------------------------------------------------------------------------//
	//判别函数的构建
	discriminant_parameter_1 = 0.627606 * Image_deviation + 10.094088 * Image_grad - 39.718507 * asm_horison
								+ 5.773537 * eng_horison - 30.390587 * con_horison + 316.283127 * idm_horison;
	discriminant_parameter_2 = -1.585143*Image_deviation - 2.686873*Image_grad + 80.200081*asm_horison
								+ 27.439622*eng_horison - 6.348111*con_horison + 61.904847*idm_horison;
	discriminant_parameter_3 = 0.250394*Image_deviation + 11.26573*Image_grad - 315.752128*asm_horison
								- 59.355475*eng_horison + 29.807364*con_horison + 222.833513*idm_horison;
	discriminant_parameter_4 = 0.056979*Image_deviation + 1.784137*Image_grad + 31.203688*asm_horison
								+ 0.020408*eng_horison + 18.381462*con_horison + 95.52347*idm_horison;
	discriminant_parameter_5 = -0.315652*Image_deviation + 0.775676*Image_grad + 203.313701*asm_horison
								+ 26.862061*eng_horison - 39.377623*con_horison - 94.106564*idm_horison;
	discriminant_parameter_6 = -1.369669*Image_deviation + 10.080256*Image_grad + 147.9757*asm_horison
								+ 67.496449*eng_horison + 53.197855*con_horison + 650.784764*idm_horison;
	discriminant_parameter_7 = -0.640182*Image_deviation + 10.387531*Image_grad + 82.377634*asm_horison
								+ 33.718542*eng_horison + 40.8465*con_horison + 396.630615*idm_horison;

	//--------------------------------------------------------------------------------//

	//Caculation4
	feature_computer(dst_vertical, asm_horison, eng_horison, con_horison, idm_horison);
	//--------------------------------------------------------------------------------//
	discriminant_parameter_1 = discriminant_parameter_1 - 1.874691 * con_horison;
	discriminant_parameter_2 = discriminant_parameter_2 + 11.4291 * con_horison;
	discriminant_parameter_3 = discriminant_parameter_3 - 31.785291 * con_horison;
	discriminant_parameter_4 = discriminant_parameter_4 - 5.533872 * con_horison;
	discriminant_parameter_5 = discriminant_parameter_5 + 6.943979 * con_horison;
	discriminant_parameter_6 = discriminant_parameter_6 - 34.910748 * con_horison;
	discriminant_parameter_7 = discriminant_parameter_7 - 52.453443 * con_horison;
	//--------------------------------------------------------------------------------//
	
	//Caculation5
	feature_computer(dst_45, asm_horison, eng_horison, con_horison, idm_horison);
	//--------------------------------------------------------------------------------//
	discriminant_parameter_1 = discriminant_parameter_1 - 0.512735 * con_horison - 356.789403;
	discriminant_parameter_2 = discriminant_parameter_2 + 1.784037 * con_horison - 69.920121;
	discriminant_parameter_3 = discriminant_parameter_3 + 0.651169 * con_horison - 53.013956;
	discriminant_parameter_4 = discriminant_parameter_4 - 0.710237 * con_horison - 100.568661;
	discriminant_parameter_5 = discriminant_parameter_5 - 0.784768 * con_horison - 3.260508;
	discriminant_parameter_6 = discriminant_parameter_6 + 0.83537 * con_horison - 753.635036;
	discriminant_parameter_7 = discriminant_parameter_7 + 9.010777 * con_horison - 467.661443;
	//--------------------------------------------------------------------------------//
	
	/*
	QString * pk = new QString;
	*pk = (*pk).setNum(discriminant_parameter_1);
	ui->label0_0->setText(*pk);
	delete pk;
	//cout << "discriminant_parameter_1:" << discriminant_parameter_1 << endl;
	*/

	//通过构建的的函数计算缺陷类型的概率累加
	double *abs_value = new double;

	//分类1000
    *abs_value = abs(2.559913 - discriminant_parameter_1);
	if (*abs_value < 1.45)
		Scratches_Rate = Scratches_Rate + 40 * 1;//40分，100%

	*abs_value = abs(-5.251672 - discriminant_parameter_2);
	if (*abs_value < 2.2)
		Scratches_Rate = Scratches_Rate + 30 * 0.5714;//30分，57.14%

	*abs_value = abs(-0.988691 - discriminant_parameter_3);
	if (*abs_value < 0.8)
		Scratches_Rate = Scratches_Rate + 25 * 0.3478;//25分，34.78%

	*abs_value = abs(-1.937991 - discriminant_parameter_4);
	if (*abs_value < 0.9)
		Scratches_Rate = Scratches_Rate + 25 * 0.4444;//25分，34.78%

	*abs_value = abs(0.709310 - discriminant_parameter_5);
	if (*abs_value < 0.9)
		Scratches_Rate = Scratches_Rate + 25 * 0.2581;//25分，25.81%

	*abs_value = abs(0.302223 - discriminant_parameter_6);
	if (*abs_value < 0.6)
		Scratches_Rate = Scratches_Rate + 25 * 0.2308;//25分，23.08%

	*abs_value = abs(0.011046 - discriminant_parameter_7);
	if (*abs_value < 1.2)
		Scratches_Rate = Scratches_Rate + 25 * 0.1346;//25分，13.46%


	//分类2000
	*abs_value = abs(-8.116260 - discriminant_parameter_1);
	if (*abs_value < 2)
		Concentrated_Rate = Concentrated_Rate + 40 * 0.6667;//40分

	*abs_value = abs(0.512635 - discriminant_parameter_2);
	if (*abs_value < 1)
		Concentrated_Rate = Concentrated_Rate + 30 * 0.2727;//30分

	*abs_value = abs(3.596209 - discriminant_parameter_3);
	if (*abs_value < 0.8)
		Concentrated_Rate = Concentrated_Rate + 25 * 0.4;//25分

	*abs_value = abs(-0.019383 - discriminant_parameter_4);
	if (*abs_value < 0.6)
		Concentrated_Rate = Concentrated_Rate + 25 * 0.2105;//25分

	*abs_value = abs(1.514468 - discriminant_parameter_5);
	if (*abs_value < 0.6)
		Concentrated_Rate = Concentrated_Rate + 25 * 0.2667;//25分

	*abs_value = abs(-0.804052 - discriminant_parameter_6);
	if (*abs_value < 0.5)
		Concentrated_Rate = Concentrated_Rate + 25 * 0.15;//25分

	*abs_value = abs(-0.174549 - discriminant_parameter_7);
	if (*abs_value < 0.3)
		Concentrated_Rate = Concentrated_Rate + 25 * 0.2667;//25分



	//分类3000
	*abs_value = abs(-1.972091 - discriminant_parameter_1);
	if (*abs_value < 0.8)
		Fold_Rate = Fold_Rate + 40 * 0.4444;//40分

	*abs_value = abs(-1.694861 - discriminant_parameter_2);
	if (*abs_value < 2.2)
		Fold_Rate = Fold_Rate + 30 * 0.2857;//30分

	*abs_value = abs(-1.426818 - discriminant_parameter_3);
	if (*abs_value < 1.2)
		Fold_Rate = Fold_Rate + 25 * 0.3077;//25分

	*abs_value = abs(2.599238 - discriminant_parameter_4);
	if (*abs_value < 2.4)
		Fold_Rate = Fold_Rate + 25 * 0.3478;//25分

	*abs_value = abs(0.523190 - discriminant_parameter_5);
	if (*abs_value < 2)
		Fold_Rate = Fold_Rate + 25 * 0.1455;//25分

	*abs_value = abs(0.500218 - discriminant_parameter_6);
	if (*abs_value < 0.65)
		Fold_Rate = Fold_Rate + 25 * 0.3077;//25分

	*abs_value = abs(0.042449 - discriminant_parameter_7);
	if (*abs_value < 0.7)
		Fold_Rate = Fold_Rate + 25 * 0.2286;//25分


	//分类4000
	*abs_value = abs(6.387809 - discriminant_parameter_1);
	if (*abs_value < 1)
		Cleavage_Rate = Cleavage_Rate + 40 * 0.6154;//40分

	*abs_value = abs(-0.488465 - discriminant_parameter_2);
	if (*abs_value < 2.4)
		Cleavage_Rate = Cleavage_Rate + 30 * 0.3077;//30分

	*abs_value = abs(-0.796352 - discriminant_parameter_3);
	if (*abs_value < 1)
		Cleavage_Rate = Cleavage_Rate + 25 * 0.3077;//25分

	*abs_value = abs(0.684365 - discriminant_parameter_4);
	if (*abs_value < 1.2 && *abs_value >0.3)
		Cleavage_Rate = Cleavage_Rate + 25 * 0.4;//25分

	*abs_value = abs(-0.633899 - discriminant_parameter_5);
	if (*abs_value < 2.2)
		Cleavage_Rate = Cleavage_Rate + 25 * 0.1404;//25分

	*abs_value = abs(-0.310549 - discriminant_parameter_6);
	if (*abs_value < 1.5)
		Cleavage_Rate = Cleavage_Rate + 25 * 0.1228;//25分

	*abs_value = abs(-0.210099 - discriminant_parameter_7);
	if (*abs_value < 0.8)
		Cleavage_Rate = Cleavage_Rate + 25 * 0.1228;//25分


	 //分类5000
	*abs_value = abs(-2.171123 - discriminant_parameter_1);
	if (*abs_value < 0.7)
		Twist_Rate = Twist_Rate + 40 * 0.4706;//40分

	*abs_value = abs(-2.867632 - discriminant_parameter_2);
	if (*abs_value < 1.3)
		Twist_Rate = Twist_Rate + 30 * 0.4667;//30分

	*abs_value = abs(0.355089 - discriminant_parameter_3);
	if (*abs_value < 0.7)
		Twist_Rate = Twist_Rate + 25 * 0.6154;//25分

	*abs_value = abs(0.163475 - discriminant_parameter_4);
	if (*abs_value < 2.2)
		Twist_Rate = Twist_Rate + 25 * 0.153846;//25分

	*abs_value = abs(-0.871723 - discriminant_parameter_5);
	if (*abs_value < 1)
		Twist_Rate = Twist_Rate + 25 * 0.2857;//25分

	*abs_value = abs(-0.789585 - discriminant_parameter_6);
	if (*abs_value < 1.6)
		Twist_Rate = Twist_Rate + 25 * 0.32;//25分

	*abs_value = abs(0.159564 - discriminant_parameter_7);
	if (*abs_value < 1.2)
		Twist_Rate = Twist_Rate + 25 * 0.1509;//25分


	//分类6000
	*abs_value = abs(6.678275 - discriminant_parameter_1);
	if (*abs_value < 1.4)
		Twist_Rate = Twist_Rate + 40 * 0.5;//40分

	*abs_value = abs(3.800330 - discriminant_parameter_2);
	if (*abs_value < 1.4)
		Twist_Rate = Twist_Rate + 30 * 0.6154;//30分

	*abs_value = abs(3.208583 - discriminant_parameter_3);
	if (*abs_value < 2.4)
		Twist_Rate = Twist_Rate + 25 * 0.4;//25分

	*abs_value = abs(0.062477 - discriminant_parameter_4);
	if (*abs_value < 1.1)
		Twist_Rate = Twist_Rate + 25 * 0.25;//25分

	*abs_value = abs(0.372976 - discriminant_parameter_5);
	if (*abs_value < 1)
		Twist_Rate = Twist_Rate + 25 * 0.2286;//25分

	*abs_value = abs(0.117933 - discriminant_parameter_6);
	if (*abs_value < 1.3)
		Twist_Rate = Twist_Rate + 25 * 0.1702;//25分

	*abs_value = abs(0.117377 - discriminant_parameter_7);
	if (*abs_value < 1.2)
		Twist_Rate = Twist_Rate + 25 * 0.1481;//25分


	//分类7000
	*abs_value = abs(-2.170779 - discriminant_parameter_1);
	if (*abs_value < 2.1)
		Fold_Rate = Fold_Rate + 40 * 0.3077;//40分

	*abs_value = abs(5.355639 - discriminant_parameter_2);
	if (*abs_value < 8)
		Fold_Rate = Fold_Rate + 30 * 0.3810;//30分

	*abs_value = abs(-4.071969 - discriminant_parameter_3);
	if (*abs_value < 2)
		Fold_Rate = Fold_Rate + 25 * 0.8;//25分

	*abs_value = abs(-0.853501 - discriminant_parameter_4);
	if (*abs_value < 1.1)
		Fold_Rate = Fold_Rate + 25 * 0.25;//25分

	*abs_value = abs(0.170171 - discriminant_parameter_5);
	if (*abs_value < 1.8)
		Fold_Rate = Fold_Rate + 25 * 0.1538;//25分

	*abs_value = abs(-0.165955 - discriminant_parameter_6);
	if (*abs_value < 1.4)
		Fold_Rate = Fold_Rate + 25 * 0.16;//25分

	*abs_value = abs(0.025307 - discriminant_parameter_7);
	if (*abs_value < 2.4)
		Fold_Rate = Fold_Rate + 25 * 0.1404;//25分

	/*
	//分类8000
	*abs_value = abs(-5.253874 - discriminant_parameter_1);
	if (*abs_value < 2.2)
		Gap_Rate = Gap_Rate + 40 * 0.7273;//40分

	*abs_value = abs(0.890344 - discriminant_parameter_2);
	if (*abs_value < 8)
		Gap_Rate = Gap_Rate + 30 * 0.1356;//30分

	*abs_value = abs(1.922053 - discriminant_parameter_3);
	if (*abs_value < 2.4)
		Gap_Rate = Gap_Rate + 25 * 0.2667;//25分

	*abs_value = abs(-0.708371 - discriminant_parameter_4);
	if (*abs_value < 1.5)
		Gap_Rate = Gap_Rate + 25 * 0.1818;//25分

	*abs_value = abs(-1.027259 - discriminant_parameter_5);
	if (*abs_value < 2.8)
		Gap_Rate = Gap_Rate + 25 * 0.1404;//25分

	*abs_value = abs(0.747741 - discriminant_parameter_6);
	if (*abs_value < 3)
		Gap_Rate = Gap_Rate + 25 * 0.1333;//25分

	*abs_value = abs(-0.058369 - discriminant_parameter_7);
	if (*abs_value < 2.8)
		Gap_Rate = Gap_Rate + 25 * 0.1333;//25分
	*/

	delete abs_value;
	
}