/*
 * gradmap.h
 *
 *  Created on: 2017-4-19
 *      Author: di
 */

#ifndef GET_GRADMAP_HPP_
#define GET_GRADMAP_HPP_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Gradmap{
public:

	Gradmap(cv::Mat X,int binnum){
		this->binnum=binnum;
		const int qori_size[3]={X.rows,X.cols,binnum};
		qori.create(3,qori_size,CV_32F);
		ori.create(X.rows,X.cols,CV_32F);
		mag.create(X.rows,X.cols,CV_32F);
		get_gradmap(X,qori,ori,mag);
	}

	cv::Mat getQori();
	cv::Mat getOri();
	cv::Mat getMag();

private:
	cv::Mat qori;
	cv::Mat ori;
	cv::Mat mag;
	int binnum;

	void get_gradmap(cv::Mat X,cv::Mat &qori,cv::Mat &ori,cv::Mat &mag);
	void imfilter2(cv::Mat src_image,cv::Mat filter,cv::Mat &dst_image);
};

#endif /* GET_GRADMAP_HPP_ */
