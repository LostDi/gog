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
	int h,w,numchannel;

	CreateCorr(cv::Mat F){
		this->F=F;
		h=F.size[0];
		w=F.size[1];
		numchannel=F.size[2];

	}


unsigned creCorr(int th_id,float Y[]){
	int starty,incy;
	starty=th_id;
	incy=threadnum;

	for(int y = starty; y<h; y = y + incy){
        for(int x = 0; x<w; x++){
            int ind = 0;
            for(int ch1 = 0; ch1<numchannel; ch1++){
                for(int ch2 = ch1; ch2<numchannel; ch2++){
                    Y[w*numchannel*y + numchannel*x + ind] = F.at<float>(y,x,ch1)*F.at<float>(y,x,ch2);
                    ind++;
                }
            }
        }
    }

    return 0;
}

unsigned creIH(int th_id,float IH[]){
	int startc,incc;
	startc=th_id;
	incc=threadnum;
	int h2=h+1;
	int w2=w+1;

	for(int ch = startc; ch< numchannel; ch = ch + incc){
	  // x = 0, y =0
	  IH[ 1 + h2 + ch*w2*h2] = F.at<float>(0,0,ch);

	  // y = 0
	  for(int x = 1; x<w; x++)
		  IH[ 1+ (x+1)*h2 + ch*w2*h2] = IH[ 1 + x*h2 + ch*w2*h2] + F.at<float>(0,x,ch);

	  for(int y = 1; y<h; y++){
		  IH[ y+1 + h2 +  ch*w2*h2] = IH[ y + h2 + ch*w2*h2] +  F.at<float>(y,0,ch); //x = 0
		  for(int x = 1; x <w; x++)
			  IH[ y+1 + (x+1)*h2 + ch*w2*h2] = IH[ y+1 + x*h2 + ch*w2*h2 ] +  IH[ y + (x+1)*h2 + ch*w2*h2] +  F.at<float>(y,x,ch)- IH[ y + x*h2 + ch*w2*h2];
	  }
	}

	return 0;
}

void getCorr(float Y[]){
	std::thread thrd[threadnum];

	for(int t=0;t<threadnum;++t){
		thrd[t]=std::thread(&CreateCorr::creCorr,this,t,std::ref(Y));
	}
	for(int t=0;t<threadnum;++t){
		thrd[t].join();
	}
	return;
}

void getCorrIH(float IH[]){
	std::thread thrd[threadnum];

	for(int t=0;t<threadnum;++t){
		thrd[t]=std::thread(&CreateCorr::creIH,this,t,std::ref(IH));
	}
	for(int t=0;t<threadnum;++t){
		thrd[t].join();
	}
	return;
}

};




#endif /* MCORR_HPP_ */
