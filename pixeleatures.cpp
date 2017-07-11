#include "pixelfeatures.h"
#include<iostream>
cv::Mat Pixelfeatures::get_pixelfeatures(cv::Mat X){

	const int F_size[3]={X.rows,X.cols,lfparam.num_element};
	cv::Mat F=cv::Mat::zeros(3,F_size,CV_32F);
	int dimstart=0;

	// y
	if(lfparam.usebase[0]){
		cv::Mat PY(1,X.cols,CV_32F,cv::Scalar::all(1.f/X.rows));
		for(int i=2;i<=X.rows;++i){
			cv::Mat temp(1,X.cols,CV_32F,cv::Scalar::all(i*1.f/X.rows));
			PY.push_back(temp);
		}

		for(int i=0;i<F.size[0];++i){
			for(int j=0;j<F.size[1];++j){
				F.at<float>(i,j,dimstart)=PY.at<float>(i,j);
			}
		}
		++dimstart;
	}

	// M_theta
	if(lfparam.usebase[1]){
		int binhog=4;
		cv::Mat img_ycbcr;
		cv::cvtColor(X,img_ycbcr,cv::COLOR_BGR2YCrCb);

		cv::Mat img_ycbcr_0(img_ycbcr.rows,img_ycbcr.cols,CV_32F);
		for(int i=0;i<img_ycbcr.rows;++i){
			for(int j=0;j<img_ycbcr.cols;++j){
				img_ycbcr.at<cv::Vec3f>(i,j)[0]=16+219*(img_ycbcr.at<cv::Vec3f>(i,j)[0])/255;
				img_ycbcr_0.at<float>(i,j)=(img_ycbcr.at<cv::Vec3f>(i,j)[0]-16)/235;
			}
		}

		Gradmap unnamed=Gradmap(img_ycbcr_0,binhog);
		cv::Mat qori=unnamed.getQori();
		cv::Mat mag=unnamed.getMag();

		for(int i=0;i<F.size[0];++i){
			for(int j=0;j<F.size[1];++j){
				for(int bin=0;bin<binhog;++bin){
					F.at<float>(i,j,bin+dimstart)=qori.at<float>(i,j,bin)*mag.at<float>(i,j);
				}
			}
		}

		dimstart=dimstart+binhog;
	}

	// RGB **the image is read by BGR. Hence, the index has changed in here**
	if(lfparam.usebase[2]){
		for(int i=0;i<F.size[0];++i){
			for(int j=0;j<F.size[1];++j){
				F.at<float>(i,j,dimstart)=X.at<cv::Vec3f>(i,j)[2]/255; //R
				F.at<float>(i,j,dimstart+1)=X.at<cv::Vec3f>(i,j)[1]/255; //G
				F.at<float>(i,j,dimstart+2)=X.at<cv::Vec3f>(i,j)[0]/255; //B
			}
		}

		dimstart=dimstart+3;
	}



	// Lab
	if(lfparam.usebase[3]){
		cv::Mat img_lab;
		cv::cvtColor(X,img_lab,cv::COLOR_BGR2Lab);

		for(int i=0;i<img_lab.rows;++i){
			for(int j=0;j<img_lab.cols;++j){
				F.at<float>(i,j,dimstart)=img_lab.at<cv::Vec3f>(i,j)[0]/100;
				F.at<float>(i,j,dimstart+1)=(img_lab.at<cv::Vec3f>(i,j)[1]+50)/255;
				F.at<float>(i,j,dimstart+2)=(img_lab.at<cv::Vec3f>(i,j)[2]+50)/255;
			}
		}
		dimstart=dimstart+3;
	}

	// HSV
	if(lfparam.usebase[4]){
		cv::Mat img_hsv;
		cv::cvtColor(X,img_hsv,cv::COLOR_BGR2HSV);

		for(int i=0;i<img_hsv.rows;++i){
			for(int j=0;j<img_hsv.cols;++j){
				F.at<float>(i,j,dimstart)=img_hsv.at<cv::Vec3f>(i,j)[0];
				F.at<float>(i,j,dimstart+1)=img_hsv.at<cv::Vec3f>(i,j)[1];
				F.at<float>(i,j,dimstart+2)=img_hsv.at<cv::Vec3f>(i,j)[2];
			}
		}
		dimstart=dimstart+3;
	}

	//nRnG
	if(lfparam.usebase[5]){

		for(int i=0;i<F.size[0];++i){
			for(int j=0;j<F.size[1];++j){
				F.at<float>(i,j,dimstart)=X.at<cv::Vec3f>(i,j)[2]/(X.at<cv::Vec3f>(i,j)[0]+X.at<cv::Vec3f>(i,j)[1]+X.at<cv::Vec3f>(i,j)[2]);
				F.at<float>(i,j,dimstart+1)=X.at<cv::Vec3f>(i,j)[1]/(X.at<cv::Vec3f>(i,j)[0]+X.at<cv::Vec3f>(i,j)[1]+X.at<cv::Vec3f>(i,j)[2]);
			}
		}
		dimstart=dimstart+2;
	}

	return F;
}
