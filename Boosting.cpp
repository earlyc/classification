#include <cv.h>
#include <ml.h>
#include <stdio.h>

using namespace cv;

#define TRAINING_SIZE 8672
#define TESTING_SIZE 8671
#define NUMBER_OF_ATTRIBUTES 10
#define NUMBER_OF_CLASSES 2

int readFromCSV(const char* filename, Mat data, Mat classes, int n_samples );

int main( int argc, char** argv )
{
	
	if(argc != 3) {
		printf("Usage: %s TRAINING_FILE.csv TEST_FILE.csv\n", argv[0]);
		return -1;
	}
	
	printf ("Using OpenCV version %s\n", CV_VERSION);
	
	clock_t t1,t2,t3;
	t1=clock();
	
	Mat trainingData = Mat(TRAINING_SIZE, NUMBER_OF_ATTRIBUTES, CV_32FC1);
	Mat trainingClasses = Mat(TRAINING_SIZE, 1, CV_32FC1);

	Mat testingData = Mat(TESTING_SIZE, NUMBER_OF_ATTRIBUTES, CV_32FC1);
	Mat testingClasses = Mat(TESTING_SIZE, 1, CV_32FC1);

	Mat varType = Mat(NUMBER_OF_ATTRIBUTES + 1, 1, CV_8U );
	varType.setTo(Scalar(CV_VAR_NUMERICAL) );

	varType.at<uchar>(NUMBER_OF_ATTRIBUTES, 0) = CV_VAR_CATEGORICAL;

	double result;

	if (readFromCSV(argv[1], trainingData, trainingClasses, TRAINING_SIZE) &&
	   readFromCSV(argv[2], testingData, testingClasses, TESTING_SIZE))
	{
		printf( "\nUsing training file: %s\n", argv[1]);
		
		CvBoost* boostTree = new CvBoost;
		
        	float priors[] = {1,1};
        	CvBoostParams params = CvBoostParams(CvBoost::REAL,  // boosting type
                                             100,			 // number of weak classifiers
                                             0.95,   		 // trim rate
                                             25, 	  		 // max depth of trees
                                             false,  		 // compute surrogate split, no missing data
                                             priors );

        	params.max_categories = 15;
        	params.min_sample_count = 5;
        	params.cv_folds = 1;
        	params.use_1se_rule = false;
        	params.truncate_pruned_tree = false;
        	params.regression_accuracy = 0.0;
		
		boostTree->train( trainingData, CV_ROW_SAMPLE, trainingClasses, Mat(), Mat(), varType, Mat(), params, false);

		t2=clock();

		Mat testSample;
		int correctClass = 0;
		int incorrectClass = 0;

		printf( "\nUsing testing file: %s\n", argv[2]);

		for (int tsample = 0; tsample < TESTING_SIZE; tsample++)
		{

			testSample = testingData.row(tsample);

			result = boostTree->predict(testSample, Mat());

			//printf("Sample %i -> class is (%d)\n", tsample, (int) result);

			if (fabs(result - testingClasses.at<float>(tsample, 0)) >= FLT_EPSILON) incorrectClass++;
			else correctClass++;
		}

		printf( "\nTest Results:\n"
		"\tCorrect classifications: %d (%g%%)\n"
		"\tIncorrect classifications: %d (%g%%)\n",
		correctClass, (double) correctClass*100/TESTING_SIZE,
		incorrectClass, (double) incorrectClass*100/TESTING_SIZE);

		t3=clock();
		float diff1 ((float)t2-(float)t1);
		float diff2 ((float)t3-(float)t2);
		float seconds1 = diff1 / CLOCKS_PER_SEC;
		float seconds2 = diff2 / CLOCKS_PER_SEC;
		printf("\tModeling: %f sec.\n", seconds1);
		printf("\tTesting: %f sec.\n", seconds2);
		
		boostTree->save("trainedBoost.xml", "boostTree");
		
		return 0;
	}

	return -1;
	
}

int readFromCSV(const char* filename, Mat data, Mat classes, int n_samples )
{
	float tmp;

	FILE* f = fopen( filename, "r" );
	if( !f )
	{
	   printf("ERROR opening file: %s\n",  filename);
	   return 0;
	}

	for(int line = 0; line < n_samples; line++)
	{

	   for(int attribute = 0; attribute < (NUMBER_OF_ATTRIBUTES + 1); attribute++)
	   {
		  if (attribute < 10)
		  {

			 fscanf(f, "%f,", &tmp);
			 data.at<float>(line, attribute) = tmp;
			 //printf("%f,", data.at<float>(line, attribute));

		  }
		  else if (attribute == 10)
		  {

			 fscanf(f, "%f,", &tmp);
			 classes.at<float>(line, 0) = tmp;
			 //printf("%f\n", classes.at<float>(line, 0));

		  }
	   }
	}

	fclose(f);

	return 1;
}