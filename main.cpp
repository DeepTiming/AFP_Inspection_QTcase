/*      ����������AFP Automated Inspection System
		�����������ò���ϵͳ�� Windows 10 64bit
		������������IDE�汾��Visual Studio Enterprise 2017
		������������OpenCV�汾��	3.4.1
		������������OpenCV_Contrib�汾  3.4.1
		������������QT�汾 5.9
		2018��5��13�� Created by  @QiaoLei
		2018��5��23�� Revised by  @QiaoLei
		1.��������λ�ã������������С
		2.���Ԥ���ϱ߽�ֽ��ߣ�����߽�Ŀ��
		3.����Ƿ���̼��ά�߽磬����
		4.����Ƿ����Ԥ�������Ӵ������У����㳤�����ޣ���ʾ��
		���case1.1����8.1������ͼƬ
*/

#include "Detection.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	Detection w;
	w.show();
	return a.exec();
}
