#include "VMatcher.h"

#define ImageTest // VideoTest or ImageTest

int main()
{
	// set parameters
	int numKeyPoints = 2000;

	//Instantiate robust matcher
	VMatcher vMatcher;

	//instantiate detector, extractor, matcher
	cv::Ptr<cv::FeatureDetector> detector = new cv::ORB(numKeyPoints);
	cv::Ptr<cv::DescriptorExtractor> extractor = new cv::ORB;
	cv::Ptr<cv::DescriptorMatcher> matcher = new cv::BruteForceMatcher<cv::HammingLUT>;

	vMatcher.setFeatureDetector(detector);
	vMatcher.setDescriptorExtractor(extractor);
	vMatcher.setDescriptorMatcher(matcher);

#ifdef ImageTest
	//Load input image detect keypoints
	cv::Mat img1;
	std::vector<cv::KeyPoint> img1_keypoints;
	cv::Mat img1_descriptors;
	cv::Mat img2;
	std::vector<cv::KeyPoint> img2_keypoints;
	cv::Mat img2_descriptors;

	std::vector<cv::DMatch>  matches;

	img1 = cv::imread("./images/1.jpg");
	img2 = cv::imread("./images/2.jpg");
	
	vMatcher.match(img1, img2, matches, img1_keypoints, img2_keypoints);
	std::cout << "matched number :" << matches.size() << std::endl;
	vMatcher.showImage(img1, img2, matches, img1_keypoints, img2_keypoints);
	cvWaitKey(0);
	system("pause");
#endif // ImageTest

#ifdef VideoTest
	cv::Mat img1;
	std::vector<cv::KeyPoint> img1_keypoints;
	cv::Mat img1_descriptors;
	img1 = cv::imread("./images/1.jpg");
	cv::VideoCapture Camera(0);
	if (!Camera.isOpened())
		return -1;
	while (cv::waitKey(33) != 27) {
		cv::Mat img2;
		std::vector<cv::KeyPoint> img2_keypoints;
		cv::Mat img2_descriptors;

		std::vector<cv::DMatch>  matches;
		
		if (!Camera.read(img2))
			break;
		//cv::imshow("video", img2);

		vMatcher.match(img1, img2, matches, img1_keypoints, img2_keypoints);
		std::cout << "matched number :" << matches.size() << std::endl;
		vMatcher.showImage(img1, img2, matches, img1_keypoints, img2_keypoints);

		system("cls");
	}
	Camera.release();
#endif // VideoTest

}