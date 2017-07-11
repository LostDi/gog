/*
 * gog.h
 *
 *  Created on: 2017-5-10
 *      Author: di
 */

#ifndef GOG_H_
#define GOG_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "gradmap.h"
#include "pixelfeatures.h"

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

class GOG{
public:
	Param param;
	PartGrid parGrid;
	cv::Size s;

	GOG(Param param,PartGrid parGrid){
		s=cv::Size(48,128);
		this->param=param;
		this->parGrid=parGrid;
	}
	GOG(Param param){
		s=cv::Size(48,128);
		this->param=param;
	}

	cv::Mat getFeature(cv::Mat image);

protected:
	cv::Mat regionGaussian(cv::Mat F);
	cv::Mat patchGaussian(cv::Mat F);
	cv::Mat gog(cv::Mat F,cv::Mat weightmap);
	inline cv::Mat vec2Mat(cv::Mat vecMat,int dim);
	inline cv::Mat halfVec(cv::Mat fullMat);

};


#endif /* GOG_H_ */
