/*
 *  train.cpp
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

/*- Usage: ./train.o -*/

int main(int argc, char** argv_) {

	/*- In variables -*/

	int max_cat_train = 250;//12500; //problem size of dataset and available time (kmeans)
	int max_dog_train = 250;//12500;
	int vocabulary_size = 1000;
	int orb_size = 400;
	char int_str[32];
	char file [64] = "\0";

	/*- Vocabulary variables -*/

	Ptr<DescriptorExtractor > extractor(
		new OpponentColorDescriptorExtractor(
			Ptr<DescriptorExtractor>(new OrbDescriptorExtractor())
			)
		);
	Mat descriptors;
	Mat training_descriptors(1,extractor->descriptorSize(),extractor->descriptorType());
	Mat mat_cat_dog;
	OrbFeatureDetector detector(orb_size);
	vector<KeyPoint> keypoints;
	Mat img, vocabulary;

	FileStorage fs("vocabulary.yml", FileStorage::WRITE);

	vector<int> labels;
	Mat response_hist;

	Ptr<DescriptorMatcher > matcher(new cv::BFMatcher(NORM_HAMMING));
	Ptr<BOWImgDescriptorExtractor> bowide(new BOWImgDescriptorExtractor(extractor,matcher));

	/*- SVM  variables-*/

	CvSVMParams params;
    params.svm_type = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.gamma = 3;
    params.degree = 3;
	CvSVM svm;



	cout << "-------- train BOW SVM -----------" << endl;

	cout << " Get vocabulary " << endl;

	
	cout<< "Getting cats words" << endl;
	 
	for(int i=0; i< max_cat_train; i++) {
		file[0]='\0';
		strcat(file, "train/cat.");
		sprintf(int_str, "%d", i);
		strcat(file, int_str);
		strcat(file, ".jpg");
	   	img = imread(file);
	   	/*- Getting ORB descriptors -*/
	   	detector.detect(img, keypoints);
	   	extractor->compute(img, keypoints, descriptors);
	   	training_descriptors.push_back(descriptors);
	}

	cout<< "Getting dogs words" << endl;

	for(int i=0; i< max_dog_train; i++) {
		file[0]='\0';
		strcat(file, "train/dog.");
		sprintf(int_str, "%d", i);
		strcat(file, int_str);
		strcat(file, ".jpg");
	   	img = imread(file);
	   	/*- Getting ORB descriptors -*/
	   	detector.detect(img, keypoints);
	   	extractor->compute(img, keypoints, descriptors);
	   	training_descriptors.push_back(descriptors);
	   	
	}

	if (training_descriptors.type() != CV_32F){

	    training_descriptors.convertTo(training_descriptors, CV_32F);
	}
	cout << training_descriptors.size() << endl;

	/*- Clustering ORB descriptors -*/
	/*- Funtion O(n2) -> too long -*/
	cout<< "Clustering words" << endl;
	BOWKMeansTrainer bowtrainer(vocabulary_size); 
	bowtrainer.add(training_descriptors);
	/*- Cluster to Vocabulary -*/
	vocabulary = bowtrainer.cluster();

	vocabulary.convertTo(vocabulary, CV_8UC1);
	fs << "vocabulary" <<vocabulary;
    fs.release();
	cout<< "Vocabulary Saved!" << endl;

	cout<< "Train SVM" << endl;

	bowide->setVocabulary(vocabulary);
 	 
	for(int i=0; i< max_cat_train; i++) {
		file[0]='\0';
		strcat(file, "train/cat.");
		sprintf(int_str, "%d", i);
		strcat(file, int_str);
		strcat(file, ".jpg");
	   	img = imread(file);

	   	/*- Get histogram of img in vocabuary -*/
	 	detector.detect(img,keypoints);
	   	bowide->compute(img, keypoints, response_hist);
	   	
	    mat_cat_dog.push_back(response_hist);
	    labels.push_back(0);
	   
	}

	for(int i=0; i< max_dog_train; i++) {
		file[0]='\0';
		strcat(file, "train/dog.");		
		sprintf(int_str, "%d", i);
		strcat(file, int_str);
		strcat(file, ".jpg");
	   	img = imread(file);

	   	/*- Get histogram of img in vocabulary -*/
	 	detector.detect(img,keypoints);
	   	bowide->compute(img, keypoints, response_hist);
	   
	    mat_cat_dog.push_back(response_hist);
	    labels.push_back(1);
	   
	}

	Mat mat_labels = Mat(labels);

	/*- Train SVM -*/
    svm.train(mat_cat_dog, mat_labels, Mat(),Mat(),params);
    svm.save("svm.xml");

    cout<< "End train - save svm.xml" << endl;
    return 0;

}



