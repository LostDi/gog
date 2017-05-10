/*
 * gog.cpp
 *
 *  Created on: 2017-5-10
 *      Author: di
 */

#include "gog.h"
#include "pixelfeatures.h"
#include <math.h>
#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <opencv2/core/eigen.hpp>

#define PI 3.141592

cv::Mat GOG::getFeature(cv::Mat image){

	  cv::Mat rimage,rimage_64f;
	  cv::resize(image,rimage,this->s);
	  rimage.convertTo(rimage_64f,CV_32F);

	  Pixelfeatures pixeleatures(param.lfparam);
	  cv::Mat F=pixeleatures.get_pixelfeatures(rimage_64f);

	  cv::Mat M1=patchGaussian(F);
	  cv::Mat M2=regionGaussian(M1);

	  return M2;
}

cv::Mat GOG::regionGaussian(cv::Mat F){
	cv::Mat weightmap;
	cv::Mat featureAll;

	if(!this->param.ifweight){
		weightmap=cv::Mat::zeros(F.size[0],F.size[1],CV_32F);
	}
	else{
		float sigma=F.size[1]/4.f;
		float  mu=F.size[1]/2.f;
		for(int i=1;i<F.size[1]+1;++i){
			cv::Mat weightmapcol(1,F.size[0],CV_32F,cv::Scalar::all( exp(-(i-mu)*(i-mu)/(2*sigma*sigma))/(sigma*sqrt(2*PI)) ));
			weightmap.push_back(weightmapcol);
		}
		weightmap=weightmap.t();
	}

	if(param.G==7){
		this->parGrid.gheight=F.size[0]/4;
		this->parGrid.gwidth=F.size[1];
		this->parGrid.ystep=parGrid.gheight/2;
		this->parGrid.xstep=parGrid.gwidth;
	}

	int gheight2 = this->parGrid.gheight/this->param.p;
	int gwidth2 = this->parGrid.gwidth/this->param.p;
	int ystep2 = this->parGrid.ystep/this->param.p;
	int xstep2 = this->parGrid.xstep/this->param.p;

	const int region_size[3]={gheight2,gwidth2,F.size[2]};
	cv::Mat region(3,region_size,CV_32F);
	for(int prow=0;prow<=F.size[0]-gheight2;prow=prow+ystep2){
		for(int pcol=0;pcol<=F.size[1]-gwidth2;pcol=pcol+xstep2){

			for(int i=0;i<region_size[0];++i){
				for(int j=0;j<region_size[1];++j){
					for(int k=0;k<region_size[2];++k){
						region.at<float>(i,j,k)=F.at<float>(i+prow,j+pcol,k);
					}
				}
			}

			cv::Mat regionFeature=GOG::gog(region,weightmap(cv::Rect(pcol,prow,gwidth2,gheight2)));
			featureAll.push_back(regionFeature);
		}
	}

	return featureAll;
}

cv::Mat GOG::patchGaussian(cv::Mat F){
	std::vector<std::vector<std::vector<float> > > result;
	for(int prow=0;prow<F.size[0];prow=prow+param.p){
		std::vector<std::vector<float> > eachRow;
		for(int pcol=0;pcol<F.size[1];pcol=pcol+param.p){
			// for each patch
			cv::Mat u=cv::Mat(F.size[2],1,CV_32F,cv::Scalar::all(0));
			cv::Mat fi=cv::Mat(F.size[2],1,CV_32F,cv::Scalar::all(0));

			int pixNum=0;
			for(int i=std::max(prow-param.p,0);i<std::min(prow+param.k-param.p,F.size[0]);++i){
				for(int j=std::max(pcol-param.p,0);j<std::min(pcol+param.k-param.p,F.size[1]);++j){
					for(int ii=0;ii<F.size[2];++ii){
						u.at<float>(ii,0)=u.at<float>(ii,0)+F.at<float>(i,j,ii);
					}
					++pixNum;
				}
			}
			u=u.mul(1.f/pixNum);
			cv::Mat sumfi=cv::Mat(F.size[2],F.size[2],CV_32F,cv::Scalar::all(0));
			for(int i=std::max(prow-param.p,0);i<std::min(prow+param.k-param.p,F.size[0]);++i){
				for(int j=std::max(pcol-param.p,0);j<std::min(pcol+param.k-param.p,F.size[1]);++j){
					for(int ii=0;ii<F.size[2];++ii){
						fi.at<float>(ii,0)=F.at<float>(i,j,ii);
					}
					cv::Mat temp;
					cv::mulTransposed(fi,temp,false,u);
					sumfi=sumfi+temp;
				}
			}

			sumfi=sumfi.mul(1.f/(pixNum-1));
			sumfi=sumfi+param.espsilon0*std::max(0.01,cv::trace(sumfi)[0])*cv::Mat::eye(sumfi.rows,sumfi.cols,CV_32F);

			cv::Mat keyMatrix;
			cv::hconcat(sumfi+u*u.t(),u,keyMatrix);
			cv::Mat lastrow=u.clone();
			lastrow.push_back(1.f);
			keyMatrix.push_back(lastrow.t());
			keyMatrix=keyMatrix.mul(pow(cv::determinant(sumfi),-1.f/(sumfi.rows+1)));

			// logm
			Eigen::Matrix<float,-1,-1> eigMat;
			cv::cv2eigen(keyMatrix,eigMat);
			Eigen::Matrix<float,-1,-1> logm;
			logm=eigMat.log();
			cv::eigen2cv(logm,keyMatrix);

			std::vector<float> toStraight;
			for(int i=0;i<keyMatrix.rows;++i){
				toStraight.push_back(keyMatrix.at<float>(i,i));
				for(int j=i+1;j<keyMatrix.cols;++j){
					toStraight.push_back(sqrt(2.f)*keyMatrix.at<float>(i,j));
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
cv::Mat GOG::gog(cv::Mat F,cv::Mat weightmap){
	cv::Mat u=cv::Mat(F.size[2],1,CV_32F,cv::Scalar::all(0));
	cv::Mat fi=cv::Mat(F.size[2],1,CV_32F,cv::Scalar::all(0));

	for(int i=0;i<F.size[0];++i){
		for(int j=0;j<F.size[1];++j){
			for(int ii=0;ii<F.size[2];++ii){
				u.at<float>(ii,0)=u.at<float>(ii,0)+F.at<float>(i,j,ii)*weightmap.at<float>(i,j);
			}
		}
	}
	u=u.mul(1.f/cv::sum(weightmap)[0]);
	cv::Mat sumfi=cv::Mat(F.size[2],F.size[2],CV_32F,cv::Scalar::all(0));
	for(int i=0;i<F.size[0];++i){
		for(int j=0;j<F.size[1];++j){
			for(int ii=0;ii<F.size[2];++ii){
				fi.at<float>(ii,0)=F.at<float>(i,j,ii);
			}
			cv::Mat temp;
			cv::mulTransposed(fi,temp,false,u);
			sumfi=sumfi+temp.mul(weightmap.at<float>(i,j));
		}
	}

	sumfi=sumfi.mul(1.f/cv::sum(weightmap)[0]);
	sumfi=sumfi+param.espsilon0*std::max(0.01,cv::trace(sumfi)[0])*cv::Mat::eye(sumfi.rows,sumfi.cols,CV_32F);

	cv::Mat keyMatrix;
	cv::hconcat(sumfi+u*u.t(),u,keyMatrix);
	cv::Mat lastrow=u.clone();
	lastrow.push_back(1.f);
	keyMatrix.push_back(lastrow.t());
	keyMatrix=keyMatrix.mul(pow(cv::determinant(sumfi),-1.f/(sumfi.rows+1)));



	// logm
	Eigen::Matrix<float,-1,-1> eigMat;
	cv::cv2eigen(keyMatrix,eigMat);
	Eigen::Matrix<float,-1,-1> logm;
	logm=eigMat.log();
	cv::eigen2cv(logm,keyMatrix);


	std::vector<float> toStraight;
	for(int i=0;i<keyMatrix.rows;++i){
		toStraight.push_back(keyMatrix.at<float>(i,i));
		for(int j=i+1;j<keyMatrix.cols;++j){
			toStraight.push_back(sqrt(2.f)*keyMatrix.at<float>(i,j));
		}
	}

	cv::Mat Feature(toStraight,1);

	return Feature;
}
