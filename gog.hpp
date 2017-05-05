/*
 * gog.hpp
 *
 *  Created on: 2017-4-25
 *      Author: di
 */
#ifndef GOG_HPP_
#define GOG_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "pixelfeatures.h"
#include <math.h>
#include <vector>
#include <iostream>

struct Param{
	lfParam lfparam;
	int p=2;
	int k=5;
	float espsilon0=0.001;
	bool ifweight=true;
	int G=7;
};

struct PartGrid{
	int gwidth;
	int gheight;
	int ystep;
	int xstep;
};

cv::Mat gog(cv::Mat F,Param param){

	std::vector<std::vector<std::vector<float> > > result;
	for(int prow=0;prow<=F.size[0]-param.k;prow=prow+param.k-param.p){
		std::vector<std::vector<float> > eachRow;
		for(int pcol=0;pcol<=F.size[1]-param.k;pcol=pcol+param.k-param.p){
			// for each patch
			cv::Mat u=cv::Mat(F.size[2],1,CV_32F,cv::Scalar::all(0));
			cv::Mat fi=cv::Mat(F.size[2],1,CV_32F,cv::Scalar::all(0));

			for(int i=0;i<param.k;++i){
				for(int j=0;j<param.k;++j){
					for(int ii=0;ii<F.size[2];++ii){
						u.at<float>(ii,1)=u.at<float>(ii,1)+F.at<float>(i+prow,j+pcol,ii)/param.k/param.k;
					}
				}
			}
			cv::Mat sumfi=cv::Mat(F.size[2],F.size[2],CV_32F,cv::Scalar::all(0));
			for(int i=0;i<param.k;++i){
				for(int j=0;j<param.k;++j){
					for(int ii=0;ii<F.size[2];++ii){
						fi.at<float>(ii,1)=F.at<float>(i+prow,j+pcol,ii);
					}
					cv::Mat temp;
					cv::mulTransposed(fi,temp,false,u);
					sumfi=sumfi+temp;
				}
			}

			sumfi=sumfi.mul(1/(param.k*param.k-1))+param.espsilon0;
			cv::Mat keyMatrix;
			cv::hconcat(sumfi,u,keyMatrix);
			cv::Mat lastrow=u.clone();
			lastrow.push_back(1.f);
			keyMatrix.push_back(lastrow.t());
			keyMatrix=keyMatrix.mul(pow(cv::determinant(sumfi),1/(sumfi.rows+1)));

			std::vector<float> toStraight;
			for(int i=0;i<keyMatrix.rows;++i){
				toStraight.push_back(keyMatrix.at<float>(i,i));
				for(int j=i+1;j<keyMatrix.cols;++j){
					toStraight.push_back(sqrt(2)*keyMatrix.at<float>(i,j));
				}
			}
			eachRow.push_back(toStraight);
		}
		result.push_back(eachRow);
	}

	const int re_size[3]={result.size(),result[0].size(),result[0][0].size()};
	cv::Mat re(3,re_size,CV_32F,cv::Scalar::all(0));
	for(int i=0;i<re_size[0];++i){
		for(int j=0;j<re_size[1];++j){
			for(int k=0;k<re_size[2];++k){
				re.at<float>(i,j,k)=result[i][j][k];
			}
		}
	}
	return re;


}

cv::Mat gog(cv::Mat F){
	std::vector<float> result;
	// for each patch
	cv::Mat u=cv::Mat(F.size[2],1,CV_32F,cv::Scalar::all(0));
	cv::Mat fi=cv::Mat(F.size[2],1,CV_32F,cv::Scalar::all(0));

	for(int i=0;i<F.size[0];++i){
		for(int j=0;j<F.size[1];++j){
			for(int ii=0;ii<F.size[2];++ii){
				u.at<float>(ii,1)=u.at<float>(ii,1)+F.at<float>(i,j,ii)/F.size[0]/F.size[1];
			}
		}
	}
	cv::Mat sumfi=cv::Mat(F.size[2],F.size[2],CV_32F,cv::Scalar::all(0));
	for(int i=0;i<F.size[0];++i){
		for(int j=0;j<F.size[1];++j){
			for(int ii=0;ii<F.size[2];++ii){
				fi.at<float>(ii,1)=F.at<float>(i,j,ii);
			}
			cv::Mat temp;
			cv::mulTransposed(fi,temp,false,u);
			sumfi=sumfi+temp;
		}
	}

	sumfi=sumfi.mul(1/(F.size[0]*F.size[1]-1))+0.001;
	cv::Mat keyMatrix;
	cv::hconcat(sumfi,u,keyMatrix);
	cv::Mat lastrow=u.clone();
	lastrow.push_back(1.f);
	keyMatrix.push_back(lastrow.t());
	keyMatrix=keyMatrix.mul(pow(cv::determinant(sumfi),1/(sumfi.rows+1)));

	std::vector<float> toStraight;
	for(int i=0;i<keyMatrix.rows;++i){
		for(int j=i+1;j<keyMatrix.cols;++j){
			toStraight.push_back(sqrt(2)*keyMatrix.at<float>(i,j));
		}
	}
	return cv::Mat(toStraight);
}

#endif /*GOG.HPP*/
