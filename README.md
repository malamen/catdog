# catdog - Diego Chavez <malamen>

#Compile 
>$ make

#Train

Train set must be in ./train/[cat||dog].[id].jpg
>$ ./train.o

train generate vocabulary.yml (BOW vocabulary) and svm.xml (svm model)

Clustering of BOW  >>> O(n^2)

#Test

Test set must be in ./test/[id].jpg
>$ ./test_catdog.o

test_catdog generate output.csv with the prediction


#Problems

I can't train all the dataset (not enough compute time available)

#Outputs

0 : cat

1 : dog

-1 : image too small to get BOW histogram 


#Other outputs

/models/[size train set]/*


#Libraries and language

OpenCV 2.4.13

C++

#PC

Macbook pro mid 2012 (only available here)

2,5 GHz Intel Core i5

16 GB 1600 MHz DDR3

