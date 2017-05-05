/*
 * pixelfeatures.h
 *
 *  Created on: 2017-4-21
 *      Author: di
 */

#ifndef PIXELFEATURES_H_
#define PIXELFEATURES_H_

#include "gradmap.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

struct lfParam{
	int num_element;
	bool usebase[6];
};

class Pixelfeatures{
private:
	lfParam lfparam;

public:
	Pixelfeatures(lfParam lfparam){
		this->lfparam=lfparam;
	}

	cv::Mat get_pixelfeatures(cv::Mat X);
};







#endif /* PIXELFEATURES_H_ */
