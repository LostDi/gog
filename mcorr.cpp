/*
 * mcorr.cpp
 *
 *  Created on: 2017-6-12
 *      Author: di
 */
#include "mcorr.h"

template<int threadnum>
unsigned CreateCorr<threadnum>:: creCorr(int th_id){
	int starty,incy;
	starty=th_id;
	incy=threadnum;

    for(int y = starty; y<h; y = y + incy)
        for(int x = 0; x<w; x++){
            int ind = 0;
            for(int ch1 = 0; ch1<numchannel; ch1++)
                for(int ch2 = ch1; ch2<numchannel; ch2++){
                    Y[y + x*h + ind*h*w] = F.at<float>(y,x,ch1)*F.at<float>(y,x,ch2);
                    ind++;
                }
        }

    return 0;
}

