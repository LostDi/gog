/*
 * gradmap.cpp
 *
 *  Created on: 2017-4-19
 *      Author: di
 */

#include "gradmap.h"
#include <math.h>
#include <iostream>

#define PI 3.141592

//public function
	cv::Mat Gradmap::getQori(){
		return qori;
	}
	cv::Mat Gradmap::getOri(){
		return ori;
	}
	cv::Mat Gradmap::getMag(){
		return mag;
	}

// private function
void Gradmap::get_gradmap(cv::Mat X,cv::Mat &out_qori,cv::Mat &out_ori,cv::Mat &out_mag){
	cv::Mat hx=(cv::Mat_<float>(1,3)<<-1.0,0.0,1.0);
	cv::Mat hy=(cv::Mat_<float>(3,1)<<1.0,0.0,-1.0);

	cv::Mat grad_x;
	cv::Mat grad_y;
	cv::Point anchor(-1,-1);

	imfilter2(X,hx,grad_x);
	imfilter2(X,hy,grad_y);

	//ori = (atan2( grad_x, grad_y) + pi)*180/pi; % gradient orientations
	for(int i=0;i<grad_x.rows;++i){
		for(int j=0;j<grad_x.cols;++j){
			ori.at<float>(i,j)=(atan2(grad_x.at<float>(i,j),grad_y.at<float>(i,j))+PI)*180/PI;
		}
	}

	//mag = sqrt(grad_x.^2 + grad_y.^2 ); % gradient magnitude
	for(int i=0;i<grad_x.rows;++i){
			for(int j=0;j<grad_x.cols;++j){
				mag.at<float>(i,j)=sqrt(grad_x.at<float>(i,j)*grad_x.at<float>(i,j)+grad_y.at<float>(i,j)*grad_y.at<float>(i,j));
			}
		}

	float binwidth=360/binnum;
	//IND = floor( ori./binwidth);
	cv::Mat IND=ori.mul(1/binwidth);
	for(int i=0;i<IND.rows;++i){
		for(int j=0;j<IND.cols;++j){
			IND.at<float>(i,j)=floor(IND.at<float>(i,j));
		}
	}



	//ref1 = IND.*binwidth;
	cv::Mat ref1=IND.mul(binwidth);
	//ref2 = (IND + 1).*binwidth;
	cv::Mat ref2=(IND+1);
	ref2=ref2.mul(binwidth);

	cv::Mat dif1=ori-ref1;
	cv::Mat dif2=ref2-ori;

	cv::Mat weight1=dif2/(dif1+dif2);
	cv::Mat weight2=dif1/(dif1+dif2);

	//IND(IND==binnum)=0
	for(int i=0;i<IND.rows;++i){
		for(int j=0;j<IND.cols;++j){
			if(IND.at<float>(i,j)==binnum){
				IND.at<float>(i,j)=0;
			}
		}
	}

	cv::Mat IND1=IND+1;
	cv::Mat IND2=IND+2;


	//IND2(IND2==binnum+1)=1
	for(int i=0;i<IND2.rows;++i){
			for(int j=0;j<IND2.cols;++j){
				if(IND2.at<float>(i,j)==binnum+1){
					IND2.at<float>(i,j)=1;
				}
			}
		}

	for(int i=0;i<X.rows;++i){
		for(int j=0;j<X.cols;++j){
			qori.at<float>(i,j,int(IND1.at<float>(i,j))-1)=weight1.at<float>(i,j);
			qori.at<float>(i,j,int(IND2.at<float>(i,j))-1)=weight2.at<float>(i,j);
		}
	}

}

void Gradmap::imfilter2(cv::Mat src_image,cv::Mat filter,cv::Mat &dst_image){

		dst_image = cv::Mat::zeros(src_image.rows,src_image.cols,CV_64F) ;

		int start_row = int(filter.rows/2) ;
		int start_col = int(filter.cols/2) ;

		cv::Mat Mid_Matrix = cv::Mat::zeros(src_image.rows+2*start_row,src_image.cols+2*start_col,CV_32F) ;

		for (int i = 0; i < src_image.rows; i++)
		{
			for (int j = 0; j < src_image.cols; j++)
			{
				Mid_Matrix.at<float>(i+start_row,j+start_col) = src_image.at<float>(i,j);
			}
		}

		int end_row = Mid_Matrix.rows -1 - start_row ;
		int end_col = Mid_Matrix.cols -1 - start_col ;

		int filter_row = filter.rows;
		int filter_col = filter.cols;

		for (int i = start_row; i <= end_row; i++)
		{
			for (int j = start_col; j <= end_col; j++)
			{
				int tmp_row = i - start_row  ;
				int tmp_col = j - start_col  ;
				for (int m = 0; m < filter_row; m++)
				{
					for (int n = 0; n < filter_col; n++)
					{
						dst_image.at<float>(tmp_row,tmp_col) += Mid_Matrix.at<float>(tmp_row + m,tmp_col + n)*filter.at<float>(m,n) ;
					}
				}
			}
		}
		return ;
}
