/*
 * main.cpp
 *
 *  Created on: 2017-4-25
 *      Author: di
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "gog.hpp"
#include <iostream>

int main( int argc, char** argv )
{
  cv::Mat image;
  //image = cv::imread( argv[1], 1 );
  image = cv::imread("/home/di/workspace/gog/src/test.jpeg");
  cv::Mat rimage,rimage_64f;
  cv::Size s(48,128);
  cv::resize(image,rimage,s);
  rimage.convertTo(rimage_64f,CV_32F);
  if(  !image.data )//argc != 2 ||
	{
	  printf( "No image data \n" );
	  return -1;
	}

  	// 1.initial param
  	Param param1;
  	static const bool temporary[] = {true,true,true,false,false,false};
  	memcpy(param1.lfparam.usebase, temporary, sizeof temporary);
  	param1.lfparam.num_element=8;


	// 2. Pixel Feature Extraction
	Pixelfeatures pixeleatures(param1.lfparam);
	cv::Mat F=pixeleatures.get_pixelfeatures(rimage_64f);

	cv::Mat vecFea;

	PartGrid partGrid;
	if(param1.G==7){
		partGrid.gheight=F.size[0]/4;
		partGrid.gwidth=F.size[1];
		partGrid.ystep=partGrid.gheight/2;
		partGrid.xstep=partGrid.gwidth;
	}

	for(int y=0;y+partGrid.gheight<=F.size[0];y=y+partGrid.ystep){
		// 2.1. each region
		const int region_size[3]={partGrid.gheight,partGrid.gwidth,F.size[2]};
		cv::Mat region(3,region_size,CV_32F);
		for(int i=0;i<region_size[0];++i){
			for(int j=0;j<region_size[1];++j){
				for(int k=0;k<region_size[2];++k){
					region.at<float>(i,j,k)=F.at<float>(i+y,j,k);
				}
			}
		}

		cv::Mat M1=gog(region,param1);
		cv::Mat M2=gog(M1);

		vecFea.push_back(M2);
		for(int i=0;i<vecFea.rows;++i){
				std::cout<<vecFea.at<float>(i,1)<<std::endl;
			}
	}



  cv::namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
  cv::imshow( "Display Image", image );

  cv::waitKey(0);

  return 0;
}
