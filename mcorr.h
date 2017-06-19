/*
 * mcorr.h
 *
 *  Created on: 2017-6-11
 *      Author: di
 */

#ifndef MCORR_HPP_
#define MCORR_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <algorithm>
#include <vector>
#include <thread>


template<int threadnum> class CreateCorr{
public:
	cv::Mat F;
	float* Y;
	int h,w,numchannel;

	CreateCorr(cv::Mat F,float* Y){
		F=F;Y=Y;

		h=F.size[0];
		w=F.size[1];
		numchannel=F.size[2];
	}


unsigned creCorr(int th_id);

void createCorr(){
	std::thread thrd[threadnum];

	for(int t=0;t<threadnum;++t){
		thrd[t]=std::thread(std::bind(CreateCorr<threadnum>::creCorr,this),t);
	}
	for(int t=0;t<threadnum;++t){
		thrd[t].join();
	}
	return;
}

float* getCorr(){
	createCorr();
	return Y;
}

};




#endif /* MCORR_HPP_ */
