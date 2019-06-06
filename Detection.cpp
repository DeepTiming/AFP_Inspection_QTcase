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

//�򿪲����ļ�
void Detection::on_openTestFile_triggered()
{
	srcImage = cv::imread("test.bmp");
	if (!srcImage.data)
	{
		QMessageBox msgBox;
		msgBox.setText(tr("The test image is not found, you could try��1)Copy new image under the current directory, rename test.jpg. 2)Use method custom."));
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
		//   ui->label1->resize(ui->label1->pixmap()->size());//���õ�ǰ��ǩΪͼ���С
		// ui->label1->resize(img.width(),img.height());

		//this->setWidget(label1);
	}
}

//���Զ����ļ�
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
		//   ui->label1->resize(ui->label1->pixmap()->size());//���õ�ǰ��ǩΪͼ���С
		// ui->label1->resize(img.width(),img.height());

		//this->setWidget(label1);
	}
}

//��ԭ
void Detection::on_restore_triggered()
{
	// cv::flip(srcImage,dstImage,-1);
	srcImage.copyTo(restoreImage);
	srcimg = cvMat2QImage(restoreImage);
	srcimg = srcimg.scaled(ui->label1_1->size());
	ui->label1_1->setPixmap(QPixmap::fromImage(srcimg));

}

//���
void Detection::on_Clear_triggered()
{
	//�˵����ļ�=>���
	//�����ǩ1�����ݡ�setText(tr("�´���")
	ui->label1_1->setText(tr("ԭʼͼ��"));
	//�����ǩ2�����ݡ�
	ui->label1_2->setText(tr("У��ͼ��"));
	ui->label1_3->setText(tr("�߽�ͼ��"));
	ui->label1_4->setText(tr("���Ӵ�ͼ��"));
	ui->label1_5->setText(tr("���紦ͼ��"));
	ui->label1_6->setText(tr("ȱ��ͼ��"));
	ui->label2_1->setText(tr("parameter��"));
	ui->label2_2->setText(tr("Angel��"));
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

//�˳�
void Detection::on_myExit_triggered()
{
	exit(0);
}

//Mat���͵�Qimage���͵�ת��
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

//ǰ����������ͼ��У������
void Detection::on_rectification_triggered()
{
	srcImageDFT = getOptimalDFT_Correction(srcImage);//�����ԭͼ����ʾ
	QImage QsrcImageDFT;
	QsrcImageDFT = cvMat2QImage(srcImageDFT);
	ui->label1_2->clear();
	QsrcImageDFT = QsrcImageDFT.scaled(ui->label1_2->width(), ui->label1_2->height());
	ui->label1_2->setPixmap(QPixmap::fromImage(QsrcImageDFT));
	if (abs(DFT_Object_Angle) < 3)
		ui->label3_2->setText(tr("С��3��"));
	else
	{
		QString tempStr;
		ui->label3_2->setText(tempStr.setNum(DFT_Object_Angle));
	}	

	double time0 = static_cast<double>(getTickCount());
	//��ȡostu������Ӧ����ֵ
	parameter = getOstu(srcImage) * 0.65;
	QString tempStr_1;
	ui->label3_1->setText(tempStr_1.setNum(parameter));
	//cout << "The return value of getOstu is: " << parameter << endl;
	//cout << endl;

	//���Ҷ�ֵ����
	contrastStretchImage = contrastStretch(srcImage);

	area_threshold_parameter(0, 0);

	getOptimalDFT_Correction_Image = getOptimalDFT_Correction(contrastStretchImage);
	//namedWindow("getOptimalDFT_Correction_Image", WINDOW_NORMAL);
	//imshow("getOptimalDFT_Correction_Image", getOptimalDFT_Correction_Image);

	//�˲���ʹͼ���ƽ��
	blur(getOptimalDFT_Correction_Image, blurImage, Size(5, 5));

	//��ֵ��
	threshold(blurImage, thresholdImage, parameter, 255, THRESH_BINARY_INV);
	//namedWindow("thresholdImageWindow", WINDOW_NORMAL);
	//imshow("thresholdImageWindow", thresholdImage);
}

//ͼ��У��
Mat Detection::getOptimalDFT_Correction(Mat srcImage)
{
#define GRAY_THRESH 150  
#define HOUGH_VOTE 80 //�������ͶƱ��������Ҫ���ٵ����ȷ��һ��ֱ�ߣ�����lines���ڷ�������  

	Mat srcImg;
	srcImg = srcImage.clone();
	//��ȡͼ�����ĵ�����  
	Point center(srcImg.cols / 2, srcImg.rows / 2);
	ImageCenter = center;

	//ͼ������  
	//�����ı߳ߴ磬��getOptimalDFTSize()����������ɢ����Ҷ�任��DFT���ߴ�  
	//��BORDER_CONSTANT��������ͼ���ð�ɫ���հײ���  
	Mat padded;
	int opWidth = getOptimalDFTSize(srcImg.rows);
	int opHeight = getOptimalDFTSize(srcImg.cols);
	copyMakeBorder(srcImg, padded, 0, opWidth - srcImg.rows, 0, opHeight - srcImg.cols, BORDER_CONSTANT, Scalar::all(0));

	//DFT  
	//DFTҪ�ֱ����ʵ�����鲿����Ҫ�����ͼ����Ϊ�����ʵ����һ��ȫ���ͼ����Ϊ������鲿  
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat comImg;
	merge(planes, 2, comImg); //ʵ�鲿�ϲ�  
	dft(comImg, comImg);

	//���DFTͼ��  
	//һ�㶼���÷���ͼ������ʾͼ����Ҷ�ı任���������Ҷ�ף�  
	//���ȵļ��㹫ʽ��magnitude = sqrt(Re(DFT) ^ 2 + Im(DFT) ^ 2)��  
	//���ڷ��ȵı仯��Χ�ܴ󣬶�һ��ͼ�����ȷ�Χֻ��[0, 255]���������һ��Ƭ��ڣ�ֻ�м��������������Ҫ��log��������ֵ�ķ�Χ��С  
	split(comImg, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magMat = planes[0];
	magMat += Scalar::all(1);
	log(magMat, magMat);

	//dft()ֱ�ӻ�õĽ���У���Ƶ����λ���Ľǣ���Ƶ����λ���м�  
	//ϰ���ϻ��ͼ�����ĵȷݣ�����Ե���ʹ��Ƶ����λ��ͼ�����ģ�Ҳ������Ƶ��ԭ��λ������  
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

	//��Ȼ��log()��С�����ݷ�Χ������Ȼ���ܱ�֤��ֵ������[0, 255]֮��  
	//����Ҫ����normalize()�淶����[0, 1]�ڣ�����convertTo()��С��ӳ�䵽[0, 255]�ڵ����������������һ����ͨ��ͼ����  
	normalize(magMat, magMat, 0, 1, CV_MINMAX);
	Mat magImg(magMat.size(), CV_8UC1);
	magMat.convertTo(magImg, CV_8UC1, 255, 0);
	//namedWindow("magnitude", WINDOW_NORMAL);
	//imshow("magnitude", magImg);
	//imwrite("imageText_mag.jpg",magImg);  

	//Hough�任Ҫ������ͼ���Ƕ�ֵ�ģ�����Ҫ��threshold()��ͼ���ֵ��  
	threshold(magImg, magImg, GRAY_THRESH, 255, CV_THRESH_BINARY);
	//imshow("mag_binary", magImg);

	//Houghֱ�߼��  
	vector<Vec2f> lines; //float�Ͷ�ά�������ϡ�lines��  
	float pi180 = (float)CV_PI / 180; //pi180��ʾ�Ƕ���1��  
	Mat linImg(magImg.size(), CV_8UC3); //����һ����ͼ��������ʾ��������  
	HoughLines(magImg, lines, 1, pi180, HOUGH_VOTE, 0, 0); //����ֱ�߼��HoughLines(dst, lines, �ֱ���, ��λ, ͶƱ��, ����1, ����2)  

														   //Houghֱ�߼����չʾ  
	int numLines = lines.size();
	for (int l = 0; l < numLines; l++)
	{
		float rho = lines[l][0], theta = lines[l][1]; //��õ�һ��������rho��thetaֵ  
		Point pt1, pt2; //������pt1��pt2  
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


	//�ҳ�����0�ȣ�90��֮��ĵ������߽Ƕ�  
	float angel = 0;
	float piThresh = (float)CV_PI / 90; //2��  
	float pi2 = CV_PI / 2;  //pi/2����  
	for (int l = 0; l < numLines; l++)
	{
		float theta = lines[l][1];
		if (abs(theta) < piThresh || abs(theta - pi2) < piThresh)
			continue;   //�ų���ˮƽ/��ֱ��  
		else
		{
			angel = theta;
			break;
		}
	}

	//������ת�Ƕ�  
	//ͼƬ�����Ƿ��β��ܼ�����ȷ  
	if (angel < pi2)
		angel = angel - CV_PI;
	if (angel != pi2)
	{
		float angelT = srcImg.rows*tan(angel) / srcImg.cols;
		angel = atan(angelT);   //��ȡ��������ͼƬ����ȷת��  
	}
	float angelD = angel * 180 / (float)CV_PI;
	cout << "ͼƬ�Ƕ�Ϊ�� " << angelD << endl;
	DFT_Object_Angle = angelD;

	Mat dstImg;
	if (abs(angelD) < 3)
		dstImg = srcImage.clone();
	else
	{
		//ת��ͼƬ  
		Mat rotMat = getRotationMatrix2D(center, angelD, 1.0);
		//Point2f center����ʾ��ת�����ĵ�  double angle����ʾ��ת�ĽǶ� double scale��ͼ����������
		dstImg = Mat::zeros(srcImg.size(), CV_8UC1);
		cout << "Size: " << srcImg.size() << endl;
		warpAffine(srcImg, dstImg, rotMat, srcImg.size(), 1, 0, Scalar(255, 255, 255));
	}
	return dstImg;
}

//ͼ��Ҷ�ֵ����
Mat Detection::contrastStretch(Mat srcImage)
{
	Mat resultImage = srcImage.clone();//"=";"clone()";"copyTo"���ֿ�����ʽ��ǰ����ǳ�������������������  
	int nRows = resultImage.rows;
	int nCols = resultImage.cols;
	//�ж�ͼ���������  
	if (resultImage.isContinuous())
	{
		nCols = nCols * nRows;
		nRows = 1;
	}
	//ͼ��ָ�����  
	uchar *pDataMat;
	int pixMax = 0, pixMin = 255;
	//����ͼ��������Сֵ  
	for (int j = 0; j < nRows; j++)
	{
		pDataMat = resultImage.ptr<uchar>(j);//ptr<>()�õ�����һ��ָ��  
		for (int i = 0; i < nCols; i++)
		{
			if (pDataMat[i] > pixMax)
				pixMax = pDataMat[i];
			if (pDataMat[i] < pixMin)
				pixMin = pDataMat[i];
		}
	}
	//�Աȶ�����ӳ��  
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
		//��ȡ�� i��������ָ��   
		const uchar * p = in.ptr<uchar>(i);
		//�Ե�i �е�ÿ������(byte)����   
		for (int j = 0; j < cols; ++j)
		{
			histogram[int(*p++)]++;
		}
	}
	int threshold;
	long sum0 = 0, sum1 = 0; //�洢ǰ���ĻҶ��ܺͼ������Ҷ��ܺ�    
	long cnt0 = 0, cnt1 = 0; //ǰ�����ܸ������������ܸ���    
	double w0 = 0, w1 = 0; //ǰ����������ռ����ͼ��ı���    
	double u0 = 0, u1 = 0;  //ǰ����������ƽ���Ҷ�    
	double variance = 0; //�����䷽��    


	double maxVariance = 0;
	for (int i = 1; i < 256; i++) //һ�α���ÿ������    
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

	//��2���������
	MatND dstHist;       // ��cv����CvHistogram *hist = cvCreateHist
	int dims = 1;
	float hranges[] = { 0, 255 };
	const float *ranges[] = { hranges };   // ������ҪΪconst����
	int size = 256;
	int channels = 0;

	//��3������ͼ���ֱ��ͼ
	calcHist(&contrastStretchImage, 1, &channels, Mat(), dstHist, dims, &size, ranges);    // cv ����cvCalcHist
	int scale = 1;

	Mat dstImage(size * scale, size, CV_8U, Scalar(0));
	//��4����ȡ���ֵ����Сֵ
	double minValue = 0;
	double maxValue = 0;
	minMaxLoc(dstHist, &minValue, &maxValue, 0, 0);  //  ��cv���õ���cvGetMinMaxHistValue

													 //��5�����Ƴ�ֱ��ͼ
	int hpt = saturate_cast<int>(0.9 * size);
	for (int i = 0; i < 256; i++)
	{
		float binValue = dstHist.at<float>(i);           //   ע��hist����float����    ����OpenCV1.0������cvQueryHistValue_1D
		int realValue = saturate_cast<int>(binValue * hpt / maxValue);
		rectangle(dstImage, Point(i*scale, size - 1), Point((i + 1)*scale - 1, size - realValue), Scalar(255));
	}
	//namedWindow("һάֱ��ͼ", WINDOW_NORMAL);
	//imshow("һάֱ��ͼ", dstImage);

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

//���߽�
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

	//�� X�����ݶ�
	Scharr(thresholdImageboundary, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	//imshow("��Ч��ͼ�� X����Scharr", abs_grad_x);

	//��Y�����ݶ�
	Scharr(thresholdImageboundary, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	//imshow("��Ч��ͼ��Y����Scharr", abs_grad_y);

	//�ϲ��ݶ�(����)
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
		ui->label1_3->setText(tr("����⣬��ͼ�񲻴��ڱ߽�"));
		//cout << "NOT, �����򲻴��ڱ߽�" << endl;
	}
	else
	{
		//cout << "YES, ��������ڱ߽磬�����ͼʾλ��" << endl;
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

//������Ӵ�
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

	// �����ں�����
	Mat find_vertical_lines_dilateImage;
	Mat kernel1 = getStructuringElement(MORPH_RECT, Size(15, 15), Point(-1, -1));
	dilate(find_vertical_lines_openImage, find_vertical_lines_dilateImage, kernel1);
	//namedWindow("find_vertical_lines_dilateImageWindow", WINDOW_NORMAL);
	//imshow("find_vertical_lines_dilateImageWindow", find_vertical_lines_dilateImage);

	// Ѱ������
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(find_vertical_lines_dilateImage, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	// ����αƽ����� + ��ȡ���κ�Բ�α߽��
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	//vector<Point2f>center(contours.size());
	//vector<float>radius(contours.size());

	//��7���������в��֣��ƽ��߽�
	for (unsigned int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);//��ָ�����ȱƽ���������� 
		boundRect[i] = boundingRect(Mat(contours_poly[i]));//����㼯�������棨up-right�����α߽�
														   //minEnclosingCircle(contours_poly[i], center[i], radius[i]);//�Ը����� 2D�㼯��Ѱ����С����İ�ΧԲ�� 
	}

	// ��8�����ƶ�������� + ��Χ�ľ��ο�
	Mat drawing = Mat::zeros(find_vertical_lines_dilateImage.size(), CV_8UC3);
	int xx = 0;
	int yy = 0;
	int areaxxyy = 0;
	float xdistance[80];
	float ydistance[80];
	int number = 0;
	for (int unsigned i = 0; i<contours.size(); i++)
	{
		Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));//���������ɫ

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
				rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);//���ƾ���
																						 //cvRectangle���������� ͼƬ�� ���Ͻǣ� ���½ǣ� ��ɫ�� ������ϸ�� �������ͣ�������
																						 //circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);//����Բ
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
		ui->label1_4->setText("����⣬��ͼ�񲻴��ڼ�϶");
		//cout << "There is no vertical line!" << endl;
	}
	else
	{
		find_vertical_lines_flag = 1;
		QString tempStr_2;
		ui->label3_4->setText(tempStr_2.setNum(number));
		Gap_Rate = Gap_Rate + 100;
		//ui->label4_4->setText(tr("��϶"));
		//cout << "The number of vertical lines is: " << number << endl;
		//cout << "ȱ�����ͣ���϶" << endl << endl;
		QString ALLInformation;
		for (int i = 0; i < number; i++)
		{
			QString vertical_lines_math,xxx,yyy,aaa;
			QString num;
			vertical_lines_math = "          No." + num.setNum(i+1) +"�� " + xxx.setNum(xdistance[i] * 0.01) + "  * "
									 + yyy.setNum(ydistance[i] * 0.01) + " = "
									 + aaa.setNum(xdistance[i] * ydistance[i] * 0.0001);
			ALLInformation = ALLInformation.append(vertical_lines_math)+"\n";

		}
		ui->label4_1->setText(ALLInformation);
		// ��ʾЧ��ͼ����
		//namedWindow("ResultWindow", WINDOW_NORMAL);
		//imshow("ResultWindow", drawing);

		//��9��Houghֱ�߼��
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

//��⽻�紦
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
	//cout << "Findlines result(All lines) as Window: Ч��ͼ" << endl << endl;
	int rate_1 = white_point_rate(ROI_lines);
	//cout << "ǰ�ߣ�" << rate_1 << endl;

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
	//cout << "����������������" << MaxcountNumber << endl;
	float countNumber = MaxcountNumber * 0.5;
	//cout << countNumber << endl;

	//�ұ߽� �Է����кܸߵ�Ҫ�󣬾���ˮƽ
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
	//cout << "���ߣ�" << rate_2 << endl;
	float whiteRate = (float)rate_2 / (float)rate_1;
	//cout << "������" << whiteRate << endl;
	//if (whiteRate > 1.09)
	//	Cleavage_Rate = Cleavage_Rate + 30;
	//cout << "find_horizontal_lines result: " << endl << endl;
	

	//namedWindow("find_horizontal_lines_ImageWindow", WINDOW_NORMAL);
	//imshow("find_horizontal_lines_ImageWindow", find_horizontal_lines_Image);
	QImage Qfind_horizontal_lines_Image = cvMat2QImage(find_horizontal_lines_Image);
	Qfind_horizontal_lines_Image = Qfind_horizontal_lines_Image.scaled(ui->label1_5->width(), ui->label1_5->height());
	ui->label1_5->setPixmap(QPixmap::fromImage(Qfind_horizontal_lines_Image));

	//Ѱ������
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(find_horizontal_lines_Image, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	//����αƽ����� + ��ȡ���κ�Բ�α߽��
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	//�������в��֣��ƽ��߽�
	for (unsigned int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);//��ָ�����ȱƽ���������� 
		boundRect[i] = boundingRect(Mat(contours_poly[i]));//����㼯�������棨up-right�����α߽�
	}

	// ���ƶ�������� + ��Χ�ľ��ο�
	Mat drawing = Mat::zeros(find_horizontal_lines_Image.size(), CV_8UC3);
	float xdistance = 0;
	float ydistance = 0;
	float area = 0.0;
	int number = 0;

	QString ALLInformation;
	for (int unsigned i = 0; i<contours.size(); i++)
	{
		Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));//���������ɫ

		xdistance = abs(boundRect[i].br().x - boundRect[i].tl().x);
		ydistance = abs(boundRect[i].br().y - boundRect[i].tl().y);
		if (xdistance * ydistance > 3000)
		{
			number++;
			QString horizontal_lines_math, xxx, yyy, aaa, num;
			horizontal_lines_math = "          No." + num.setNum(number) + "�� " + xxx.setNum(xdistance * 0.01) + "  * "
												+ yyy.setNum(ydistance * 0.01) + " = "
												+ aaa.setNum(xdistance * ydistance * 0.0001);
			//cout << "��ͼʾ���򾭲�����ʵ�ʳ���Ϊ�� " << xdistance * 0.01 << " mm��" << endl;
			//cout << "��ͼʾ���򾭲�����ʵ�ʿ��Ϊ�� " << ydistance * 0.01 << " mm��" << endl;
			//cout << "��ͼʾ���򾭲�����ʵ�����Ϊ�� " << xdistance * ydistance * 0.0001 << " mm*mm��" << endl;
			if (ydistance > 40)
				horizontal_lines_math = horizontal_lines_math.append("\n ����ֱ���յ��ܱ�ȱ��Ӱ��ϴ󣬲ο������С��������Ҫ�������˹��۲�Ԥ���ϣ�");
				//cout << "����ֱ���յ��ܱ�ȱ��Ӱ��ϴ󣬲ο������С��������Ҫ�������˹��۲�Ԥ���ϣ�" << endl;
			ALLInformation = ALLInformation.append(horizontal_lines_math) + "\n";
			rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);//���ƾ���
			// cvRectangle���������� ͼƬ�� ���Ͻǣ� ���½ǣ� ��ɫ�� ������ϸ�� �������ͣ�������
		}
		else
			continue;
	}
	//cout << "The number of horizontal lines is: " << number << endl << endl;
	QString tempStr_3;
	ui->label3_5->setText(tempStr_3.setNum(number));
	if (number > 5)
	{
		ALLInformation = ALLInformation.append("�˴�����ֱ��״ȱ�ݣ���Ԥ���Ͻ��紦����") + "\n";
		Cleavage_Rate = Cleavage_Rate + 30;
		Twist_Rate = Twist_Rate + 30;
	}
	ui->label4_2->setText(ALLInformation);

		//cout << "�˴�����ֱ��״ȱ�ݣ���Ԥ���Ͻ��紦����" << endl;
	//cout << "find_horizontal_lines result as Window <find_horizontal_lines_ImageWindow>" << endl;
}

//�����״ȱ��
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
		// �������
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
		ui->label1_6->setText("����⣬��ͼ��û����״ȱ��");
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

		//����������͹��
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

//һ����λ����
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

	//����ȱ�ݱ���ֵ

}

//aboutme
void Detection::on_about_triggered()
{
	
		QMessageBox msgBox;
		QString Aboutme;
		Aboutme = Aboutme.append("����������AFP Automated Inspection System") + "\n";
		Aboutme = Aboutme.append("�����������ò���ϵͳ�� Windows 10 64bit") + "\n";
		Aboutme = Aboutme.append("������������IDE�汾��Visual Studio Enterprise 2017") + "\n";
		Aboutme = Aboutme.append("������������OpenCV�汾��	3.4.1") + "\n";
		Aboutme = Aboutme.append("������������OpenCV_Contrib�汾  3.4.1") + "\n";
		Aboutme = Aboutme.append("������������QT�汾  5.9") + "\n";
		Aboutme = Aboutme.append("2018��5��13�� Created by  @QiaoLei") + "\n";
		Aboutme = Aboutme.append("2018��5��29�� Revised by  @QiaoLei") + "\n";
		Aboutme = Aboutme.append("���ܼ�飺") + "\n";
		Aboutme = Aboutme.append("	1.��������λ�ã������������С") + "\n";
		Aboutme = Aboutme.append("	2.���Ԥ���ϱ߽�ֽ��ߣ�����߽�Ŀ��") + "\n";
		Aboutme = Aboutme.append("	3.����Ƿ���̼��ά�߽�") + "\n";
		Aboutme = Aboutme.append("	4.����Ƿ����Ԥ�������Ӵ������У����㳤�����ޣ���ʾ��") + "\n";
		msgBox.setText(Aboutme);
		msgBox.exec();
		
}

//�����ֵͼ���еİ׵����
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

//ͳ��ȱ�ݵı���
void Detection::on_typeaction_triggered()
{
	//�������첿��
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
		*p_rate = (*p_rate).append(tr(" ����"));
	if (Gap_Rate > 70)
		*p_rate = (*p_rate).append(tr(" ��϶"));
	if (Overlap_Rate>70)
		*p_rate = (*p_rate).append(tr(" �ص�"));
	if (Missing_Rate>70)
		*p_rate = (*p_rate).append(tr(" ȱʧ"));
	if (Twist_Rate>70)
		*p_rate = (*p_rate).append(tr(" Ťת"));
	if (Cleavage_Rate>70)
		*p_rate = (*p_rate).append(tr(" ����"));
	if (Fold_Rate>70)
		*p_rate = (*p_rate).append(tr(" ����"));
	if (Bridge_Rate>70)
		*p_rate = (*p_rate).append(tr(" ����"));
	if (Concentrated_Rate>70)
		*p_rate = (*p_rate).append(tr(" �ֲ�"));
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

//�����׼��
double Detection::cal_mean_stddev(Mat testImage) {
	//void mean_stddev_AvG(Mat testImage)
	Mat src = testImage.clone();
	Mat mat_mean, mat_stddev;
	meanStdDev(src, mat_mean, mat_stddev);
	double m, s;
	m = mat_mean.at<double>(0, 0);
	s = mat_stddev.at<double>(0, 0);
	//cout << "�ҶȾ�ֵ�ǣ�" << m << endl;
	//cout << "��׼���ǣ�" << s << endl;
	return s;
}

//�����ݶ�
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
	//cout << "ƽ���ݶ��ǣ�" << imageAvG << endl;
	return imageAvG;
}

//0�ȻҶȹ�������
void Detection::getglcm_horison(Mat& input, Mat& dst)
{
	Mat src = input;
	CV_Assert(1 == src.channels());
	src.convertTo(src, CV_32S);
	int height = src.rows;
	int width = src.cols;
	int max_gray_level = 0;
	for (int j = 0; j < height; j++)//Ѱ�����ػҶ����ֵ
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
	max_gray_level++;//���ػҶ����ֵ��1��Ϊ�þ�����ӵ�еĻҶȼ���
	if (max_gray_level > 16)//���Ҷȼ�������16����ͼ��ĻҶȼ���С��16������С�Ҷȹ�������Ĵ�С��
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
	else//���Ҷȼ���С��16����������Ӧ�ĻҶȹ�������
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

//90�ȻҶȹ�������
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
		for (int i = 0; i < height; i++)//��ͼ��ĻҶȼ���С��16������С�Ҷȹ�������Ĵ�С��
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

//45�ȻҶȹ�������
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
		for (int i = 0; i < height; i++)//��ͼ��ĻҶȼ���С��16������С�Ҷȹ�������Ĵ�С��
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

//135�ȻҶȹ�������
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
		for (int i = 0; i < height; i++)//��ͼ��ĻҶȼ���С��16������С�Ҷȹ�������Ĵ�С��
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

//��������ֵ
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
			total += srcdata[j];//��ͼ���������صĻҶ�ֵ�ĺ�
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
			copydata[j] = (double)srcdata[j] / (double)total;//ͼ��ÿһ�����صĵ�ֵ���������ܺ�
		}
	}


	for (int i = 0; i < height; i++)
	{
		double*srcdata = copy.ptr<double>(i);
		for (int j = 0; j < width; j++)
		{
			Asm += srcdata[j] * srcdata[j];//����
			if (srcdata[j]>0)
				Eng -= srcdata[j] * log(srcdata[j]);//��             
			Con += (double)(i - j)*(double)(i - j)*srcdata[j];//�Աȶ�
			Idm += srcdata[j] / (1 + (double)(i - j)*(double)(i - j));//����
		}
	}
}

//ȱ�����ͼ��
void Detection::Spss_Fisher_statics(int, void*)
{
	//�б����õĲ���
	double discriminant_parameter_1;//ռ30��
	double discriminant_parameter_2;//ռ20��
	double discriminant_parameter_3;//ռ15��
	double discriminant_parameter_4;//ռ15��
	double discriminant_parameter_5;//ռ10��
	double discriminant_parameter_6;//ռ5��
	double discriminant_parameter_7;//ռ5��

	//Caculation1
	double Image_deviation = cal_mean_stddev(contrastStretchImage);//����1
	
	//Caculation2
	double Image_grad = compute_grad(contrastStretchImage);//����2

	Mat dst_horison, dst_vertical, dst_45, dst_135;
	getglcm_horison(contrastStretchImage, dst_horison);
	getglcm_vertical(contrastStretchImage, dst_vertical);
	getglcm_45(contrastStretchImage, dst_45);
	//getglcm_135(contrastStretchImage, dst_135);

	
	//Caculation3
	double eng_horison = 0, con_horison = 0, idm_horison = 0, asm_horison = 0;
	feature_computer(dst_horison, asm_horison, eng_horison, con_horison, idm_horison);
	//--------------------------------------------------------------------------------//
	//�б����Ĺ���
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

	//ͨ�������ĵĺ�������ȱ�����͵ĸ����ۼ�
	double *abs_value = new double;

	//����1000
    *abs_value = abs(2.559913 - discriminant_parameter_1);
	if (*abs_value < 1.45)
		Scratches_Rate = Scratches_Rate + 40 * 1;//40�֣�100%

	*abs_value = abs(-5.251672 - discriminant_parameter_2);
	if (*abs_value < 2.2)
		Scratches_Rate = Scratches_Rate + 30 * 0.5714;//30�֣�57.14%

	*abs_value = abs(-0.988691 - discriminant_parameter_3);
	if (*abs_value < 0.8)
		Scratches_Rate = Scratches_Rate + 25 * 0.3478;//25�֣�34.78%

	*abs_value = abs(-1.937991 - discriminant_parameter_4);
	if (*abs_value < 0.9)
		Scratches_Rate = Scratches_Rate + 25 * 0.4444;//25�֣�34.78%

	*abs_value = abs(0.709310 - discriminant_parameter_5);
	if (*abs_value < 0.9)
		Scratches_Rate = Scratches_Rate + 25 * 0.2581;//25�֣�25.81%

	*abs_value = abs(0.302223 - discriminant_parameter_6);
	if (*abs_value < 0.6)
		Scratches_Rate = Scratches_Rate + 25 * 0.2308;//25�֣�23.08%

	*abs_value = abs(0.011046 - discriminant_parameter_7);
	if (*abs_value < 1.2)
		Scratches_Rate = Scratches_Rate + 25 * 0.1346;//25�֣�13.46%


	//����2000
	*abs_value = abs(-8.116260 - discriminant_parameter_1);
	if (*abs_value < 2)
		Concentrated_Rate = Concentrated_Rate + 40 * 0.6667;//40��

	*abs_value = abs(0.512635 - discriminant_parameter_2);
	if (*abs_value < 1)
		Concentrated_Rate = Concentrated_Rate + 30 * 0.2727;//30��

	*abs_value = abs(3.596209 - discriminant_parameter_3);
	if (*abs_value < 0.8)
		Concentrated_Rate = Concentrated_Rate + 25 * 0.4;//25��

	*abs_value = abs(-0.019383 - discriminant_parameter_4);
	if (*abs_value < 0.6)
		Concentrated_Rate = Concentrated_Rate + 25 * 0.2105;//25��

	*abs_value = abs(1.514468 - discriminant_parameter_5);
	if (*abs_value < 0.6)
		Concentrated_Rate = Concentrated_Rate + 25 * 0.2667;//25��

	*abs_value = abs(-0.804052 - discriminant_parameter_6);
	if (*abs_value < 0.5)
		Concentrated_Rate = Concentrated_Rate + 25 * 0.15;//25��

	*abs_value = abs(-0.174549 - discriminant_parameter_7);
	if (*abs_value < 0.3)
		Concentrated_Rate = Concentrated_Rate + 25 * 0.2667;//25��



	//����3000
	*abs_value = abs(-1.972091 - discriminant_parameter_1);
	if (*abs_value < 0.8)
		Fold_Rate = Fold_Rate + 40 * 0.4444;//40��

	*abs_value = abs(-1.694861 - discriminant_parameter_2);
	if (*abs_value < 2.2)
		Fold_Rate = Fold_Rate + 30 * 0.2857;//30��

	*abs_value = abs(-1.426818 - discriminant_parameter_3);
	if (*abs_value < 1.2)
		Fold_Rate = Fold_Rate + 25 * 0.3077;//25��

	*abs_value = abs(2.599238 - discriminant_parameter_4);
	if (*abs_value < 2.4)
		Fold_Rate = Fold_Rate + 25 * 0.3478;//25��

	*abs_value = abs(0.523190 - discriminant_parameter_5);
	if (*abs_value < 2)
		Fold_Rate = Fold_Rate + 25 * 0.1455;//25��

	*abs_value = abs(0.500218 - discriminant_parameter_6);
	if (*abs_value < 0.65)
		Fold_Rate = Fold_Rate + 25 * 0.3077;//25��

	*abs_value = abs(0.042449 - discriminant_parameter_7);
	if (*abs_value < 0.7)
		Fold_Rate = Fold_Rate + 25 * 0.2286;//25��


	//����4000
	*abs_value = abs(6.387809 - discriminant_parameter_1);
	if (*abs_value < 1)
		Cleavage_Rate = Cleavage_Rate + 40 * 0.6154;//40��

	*abs_value = abs(-0.488465 - discriminant_parameter_2);
	if (*abs_value < 2.4)
		Cleavage_Rate = Cleavage_Rate + 30 * 0.3077;//30��

	*abs_value = abs(-0.796352 - discriminant_parameter_3);
	if (*abs_value < 1)
		Cleavage_Rate = Cleavage_Rate + 25 * 0.3077;//25��

	*abs_value = abs(0.684365 - discriminant_parameter_4);
	if (*abs_value < 1.2 && *abs_value >0.3)
		Cleavage_Rate = Cleavage_Rate + 25 * 0.4;//25��

	*abs_value = abs(-0.633899 - discriminant_parameter_5);
	if (*abs_value < 2.2)
		Cleavage_Rate = Cleavage_Rate + 25 * 0.1404;//25��

	*abs_value = abs(-0.310549 - discriminant_parameter_6);
	if (*abs_value < 1.5)
		Cleavage_Rate = Cleavage_Rate + 25 * 0.1228;//25��

	*abs_value = abs(-0.210099 - discriminant_parameter_7);
	if (*abs_value < 0.8)
		Cleavage_Rate = Cleavage_Rate + 25 * 0.1228;//25��


	 //����5000
	*abs_value = abs(-2.171123 - discriminant_parameter_1);
	if (*abs_value < 0.7)
		Twist_Rate = Twist_Rate + 40 * 0.4706;//40��

	*abs_value = abs(-2.867632 - discriminant_parameter_2);
	if (*abs_value < 1.3)
		Twist_Rate = Twist_Rate + 30 * 0.4667;//30��

	*abs_value = abs(0.355089 - discriminant_parameter_3);
	if (*abs_value < 0.7)
		Twist_Rate = Twist_Rate + 25 * 0.6154;//25��

	*abs_value = abs(0.163475 - discriminant_parameter_4);
	if (*abs_value < 2.2)
		Twist_Rate = Twist_Rate + 25 * 0.153846;//25��

	*abs_value = abs(-0.871723 - discriminant_parameter_5);
	if (*abs_value < 1)
		Twist_Rate = Twist_Rate + 25 * 0.2857;//25��

	*abs_value = abs(-0.789585 - discriminant_parameter_6);
	if (*abs_value < 1.6)
		Twist_Rate = Twist_Rate + 25 * 0.32;//25��

	*abs_value = abs(0.159564 - discriminant_parameter_7);
	if (*abs_value < 1.2)
		Twist_Rate = Twist_Rate + 25 * 0.1509;//25��


	//����6000
	*abs_value = abs(6.678275 - discriminant_parameter_1);
	if (*abs_value < 1.4)
		Twist_Rate = Twist_Rate + 40 * 0.5;//40��

	*abs_value = abs(3.800330 - discriminant_parameter_2);
	if (*abs_value < 1.4)
		Twist_Rate = Twist_Rate + 30 * 0.6154;//30��

	*abs_value = abs(3.208583 - discriminant_parameter_3);
	if (*abs_value < 2.4)
		Twist_Rate = Twist_Rate + 25 * 0.4;//25��

	*abs_value = abs(0.062477 - discriminant_parameter_4);
	if (*abs_value < 1.1)
		Twist_Rate = Twist_Rate + 25 * 0.25;//25��

	*abs_value = abs(0.372976 - discriminant_parameter_5);
	if (*abs_value < 1)
		Twist_Rate = Twist_Rate + 25 * 0.2286;//25��

	*abs_value = abs(0.117933 - discriminant_parameter_6);
	if (*abs_value < 1.3)
		Twist_Rate = Twist_Rate + 25 * 0.1702;//25��

	*abs_value = abs(0.117377 - discriminant_parameter_7);
	if (*abs_value < 1.2)
		Twist_Rate = Twist_Rate + 25 * 0.1481;//25��


	//����7000
	*abs_value = abs(-2.170779 - discriminant_parameter_1);
	if (*abs_value < 2.1)
		Fold_Rate = Fold_Rate + 40 * 0.3077;//40��

	*abs_value = abs(5.355639 - discriminant_parameter_2);
	if (*abs_value < 8)
		Fold_Rate = Fold_Rate + 30 * 0.3810;//30��

	*abs_value = abs(-4.071969 - discriminant_parameter_3);
	if (*abs_value < 2)
		Fold_Rate = Fold_Rate + 25 * 0.8;//25��

	*abs_value = abs(-0.853501 - discriminant_parameter_4);
	if (*abs_value < 1.1)
		Fold_Rate = Fold_Rate + 25 * 0.25;//25��

	*abs_value = abs(0.170171 - discriminant_parameter_5);
	if (*abs_value < 1.8)
		Fold_Rate = Fold_Rate + 25 * 0.1538;//25��

	*abs_value = abs(-0.165955 - discriminant_parameter_6);
	if (*abs_value < 1.4)
		Fold_Rate = Fold_Rate + 25 * 0.16;//25��

	*abs_value = abs(0.025307 - discriminant_parameter_7);
	if (*abs_value < 2.4)
		Fold_Rate = Fold_Rate + 25 * 0.1404;//25��

	/*
	//����8000
	*abs_value = abs(-5.253874 - discriminant_parameter_1);
	if (*abs_value < 2.2)
		Gap_Rate = Gap_Rate + 40 * 0.7273;//40��

	*abs_value = abs(0.890344 - discriminant_parameter_2);
	if (*abs_value < 8)
		Gap_Rate = Gap_Rate + 30 * 0.1356;//30��

	*abs_value = abs(1.922053 - discriminant_parameter_3);
	if (*abs_value < 2.4)
		Gap_Rate = Gap_Rate + 25 * 0.2667;//25��

	*abs_value = abs(-0.708371 - discriminant_parameter_4);
	if (*abs_value < 1.5)
		Gap_Rate = Gap_Rate + 25 * 0.1818;//25��

	*abs_value = abs(-1.027259 - discriminant_parameter_5);
	if (*abs_value < 2.8)
		Gap_Rate = Gap_Rate + 25 * 0.1404;//25��

	*abs_value = abs(0.747741 - discriminant_parameter_6);
	if (*abs_value < 3)
		Gap_Rate = Gap_Rate + 25 * 0.1333;//25��

	*abs_value = abs(-0.058369 - discriminant_parameter_7);
	if (*abs_value < 2.8)
		Gap_Rate = Gap_Rate + 25 * 0.1333;//25��
	*/

	delete abs_value;
	
}