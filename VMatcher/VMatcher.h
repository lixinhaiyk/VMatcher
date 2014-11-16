#ifndef VMatcher
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp" 
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/legacy/legacy.hpp"
#endif // !VMatcher

class VMatcher 
{
private:
	// pointer to the feature point detector object
	cv::Ptr<cv::FeatureDetector> detector;
	// pointer to the feature descriptor extractor object
	cv::Ptr<cv::DescriptorExtractor> extractor;
	// pointer to the Matcher object
	cv::Ptr<cv::DescriptorMatcher > matcher;
	float ratio; // max ratio between 1st and 2nd NN
	bool refineF; // if true will refine the F matrix
	double distance; // min distance to epipolar
	double confidence; // confidence level (probability)

public:
	VMatcher() : ratio(0.65f), refineF(true),
		confidence(0.99), distance(3.0)
	{
			// ORB is the default feature
			detector= new cv::OrbFeatureDetector();
			extractor= new cv::OrbDescriptorExtractor();
			matcher= new cv::BruteForceMatcher<cv::HammingLUT>;
	}

	// Set the feature detector
	void setFeatureDetector(cv::Ptr<cv::FeatureDetector>& detect) 
	{
			detector= detect;
	}
	// Set the descriptor extractor
	void setDescriptorExtractor(cv::Ptr<cv::DescriptorExtractor>& desc) 
	{
			extractor= desc;
	}
	// Set the VMatcher
	void setDescriptorMatcher(cv::Ptr<cv::DescriptorMatcher>& match) 
	{
			matcher= match;
	}
	// Set confidence level
	void setConfidenceLevel(double conf) 
	{
			confidence= conf;
	}
	//Set MinDistanceToEpipolar
	void setMinDistanceToEpipolar(double dist) 
	{
		distance= dist;
	}
	//Set ratio
	void setRatio(float rat) 
	{
		ratio= rat;
	}

	int match(cv::Mat& image1, cv::Mat& image2, // input images
		// output matches and keypoints
		std::vector<cv::DMatch>& matches,
		std::vector<cv::KeyPoint>& keypoints1,
		std::vector<cv::KeyPoint>& keypoints2);

	cv::Mat ransacTest(
		const std::vector<cv::DMatch>& matches,
		const std::vector<cv::KeyPoint>& keypoints1,
		const std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::DMatch>& outMatches);

	void symmetryTest(
		const std::vector<std::vector<cv::DMatch> >& matches1,
		const std::vector<std::vector<cv::DMatch> >& matches2,
		std::vector<cv::DMatch>& symMatches);

	int ratioTest(std::vector<std::vector<cv::DMatch>>& matches);

	void showImage(cv::Mat& image1,
		cv::Mat& image2, // input images
		// output matches and keypoints
		std::vector<cv::DMatch>& matches,
		std::vector<cv::KeyPoint>& keypoints1,
		std::vector<cv::KeyPoint>& keypoints2);
};
