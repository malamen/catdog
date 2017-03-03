/*
 *  test.cpp
 *  cat-dog
 *
 *  Created by Diego Chavez - Malamen.
 *  
 *
 */

#include <stdio.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>



using namespace cv;
using namespace std;

/*- Usage: ./test_catdog.o -*/

int main(int argc, char** argv_) {

	/*- In variables -*/
	int test_size = 12500;
	int orb_size = 400;
	char file [64] = "\0";
	char int_str[32];
	vector<KeyPoint> keypoints;

	Mat img, vocabulary, response_hist;
	CvSVM svm;

	OrbFeatureDetector detector(orb_size);
	Ptr<DescriptorExtractor > extractor(
		new OpponentColorDescriptorExtractor(
			Ptr<DescriptorExtractor>(new OrbDescriptorExtractor())
			)
		);
	Ptr<DescriptorMatcher > matcher(new cv::BFMatcher(NORM_HAMMING));
	Ptr<BOWImgDescriptorExtractor> bowide(new BOWImgDescriptorExtractor(extractor,matcher));

	FileStorage fs("vocabulary.yml", FileStorage::READ);
	/*- Out variables -*/
	ofstream myfile;
	myfile.open ("output.csv");

	/*- Load svm and vocabulary -*/
	svm.load("svm.xml");
	fs["vocabulary"] >> vocabulary;
	fs.release();
	bowide->setVocabulary(vocabulary);

	for(int i=1; i<= test_size; i++) {
		myfile << i;
		myfile << ",";
		file[0]='\0';
		strcat(file, "test/");
		sprintf(int_str, "%d", i);
		strcat(file, int_str);
		strcat(file, ".jpg");
	   	img = imread(file);
	   	cout << file << endl;
	   	/*- Calculate histogram in vocabulare each img -*/
	   	detector.detect(img,keypoints);
	   	bowide->compute(img, keypoints, response_hist);

	   	if(response_hist.size()==Size(0,0))
	   		myfile << -1;

	   	/*- Predict in svm -*/
	   	else
	   		myfile << svm.predict(response_hist);
	   	myfile << "\n";

	}

	myfile.close();

	return 0;
}