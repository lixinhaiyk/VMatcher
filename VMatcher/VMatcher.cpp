#include "VMatcher.h"

int VMatcher::ratioTest(std::vector<std::vector<cv::DMatch> >
	&matches) 
{
		int removed=0;
		// for all matches
		for (std::vector<std::vector<cv::DMatch> >::iterator
			matchIterator= matches.begin();
			matchIterator!= matches.end(); ++matchIterator) 
		{
				// if 2 NN has been identified
				if (matchIterator->size() > 1) 
				{
					// check distance ratio
					if ((*matchIterator)[0].distance/
						(*matchIterator)[1].distance > ratio)
					{
							matchIterator->clear(); // remove match
							removed++;
					}
				} else 
				{ // does not have 2 neighbours
					matchIterator->clear(); // remove match
					removed++;
				}
		}
		return removed;// return the number of removed point
}

// Insert symmetrical matches in symMatches vector
void VMatcher::symmetryTest(
	const std::vector<std::vector<cv::DMatch> >& matches1,
	const std::vector<std::vector<cv::DMatch> >& matches2,
	std::vector<cv::DMatch>& symMatches)
{
		// for all matches image 1 -> image 2
		for (std::vector<std::vector<cv::DMatch> >::
			const_iterator matchIterator1= matches1.begin();
			matchIterator1!= matches1.end(); ++matchIterator1) 
		{
				// ignore deleted matches
				if (matchIterator1->size() < 2)
					continue;
				// for all matches image 2 -> image 1
				for (std::vector<std::vector<cv::DMatch> >::
					const_iterator matchIterator2= matches2.begin();
					matchIterator2!= matches2.end();
				++matchIterator2) 
				{
					// ignore deleted matches
					if (matchIterator2->size() < 2)
						continue;
					// Match symmetry test
					if ((*matchIterator1)[0].queryIdx ==
						(*matchIterator2)[0].trainIdx &&
						(*matchIterator2)[0].queryIdx ==
						(*matchIterator1)[0].trainIdx) 
					{
							// add symmetrical match
							symMatches.push_back(
								cv::DMatch((*matchIterator1)[0].queryIdx,
								(*matchIterator1)[0].trainIdx,
								(*matchIterator1)[0].distance));
							break; // next match in image 1 -> image 2
					}
				}
		}
}

// Identify good matches using RANSAC
// Return fundemental matrix
cv::Mat VMatcher::ransacTest(
	const std::vector<cv::DMatch>& matches,
	const std::vector<cv::KeyPoint>& keypoints1,
	const std::vector<cv::KeyPoint>& keypoints2,
	std::vector<cv::DMatch>& outMatches) 
{
		// Convert keypoints into Point2f
		std::vector<cv::Point2f> points1, points2;
		cv::Mat fundemental;
		for (std::vector<cv::DMatch>::
			const_iterator it= matches.begin();
			it!= matches.end(); ++it) 
		{
				// Get the position of left keypoints
				float x= keypoints1[it->queryIdx].pt.x;
				float y= keypoints1[it->queryIdx].pt.y;
				points1.push_back(cv::Point2f(x,y));
				// Get the position of right keypoints
				x= keypoints2[it->trainIdx].pt.x;
				y= keypoints2[it->trainIdx].pt.y;
				points2.push_back(cv::Point2f(x,y));
		}
		// Compute F matrix using RANSAC
		std::vector<uchar> inliers(points1.size(),0);
		if (points1.size()>0&&points2.size()>0)
		{
			cv::Mat fundemental= cv::findFundamentalMat(
				cv::Mat(points1),cv::Mat(points2), // matching points
				inliers,       // match status (inlier or outlier)
				CV_FM_RANSAC, // RANSAC method
				distance,      // distance to epipolar line
				confidence); // confidence probability
			// extract the surviving (inliers) matches
			std::vector<uchar>::const_iterator
				itIn= inliers.begin();
			std::vector<cv::DMatch>::const_iterator
				itM= matches.begin();
			// for all matches
			for ( ;itIn!= inliers.end(); ++itIn, ++itM) 
			{
				if (*itIn) 
				{ // it is a valid match
					outMatches.push_back(*itM);
				}
			}
			if (refineF) 
			{
				// The F matrix will be recomputed with
				// all accepted matches
				// Convert keypoints into Point2f
				// for final F computation
				points1.clear();
				points2.clear();
				for (std::vector<cv::DMatch>::
					const_iterator it= outMatches.begin();
					it!= outMatches.end(); ++it) 
				{
						// Get the position of left keypoints
						float x= keypoints1[it->queryIdx].pt.x;
						float y= keypoints1[it->queryIdx].pt.y;
						points1.push_back(cv::Point2f(x,y));
						// Get the position of right keypoints
						x= keypoints2[it->trainIdx].pt.x;
						y= keypoints2[it->trainIdx].pt.y;
						points2.push_back(cv::Point2f(x,y));
				}
				// Compute 8-point F from all accepted matches
				if (points1.size()>0&&points2.size()>0)
				{
					fundemental= cv::findFundamentalMat(
						cv::Mat(points1),cv::Mat(points2), // matches
						CV_FM_8POINT); // 8-point method
				}
			}
		}
	return fundemental;
}

// Match feature points using symmetry test and RANSAC
// Return the number of matched point
int VMatcher:: match(cv::Mat& image1,
	cv::Mat& image2, // input images
	// output matches and keypoints
	std::vector<cv::DMatch>& matches,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2) 
{
	// 1a. Detection of the ORB features
	detector->detect(image1,keypoints1);
	detector->detect(image2,keypoints2);
	// 1b. Extraction of the ORB descriptors
	cv::Mat descriptors1, descriptors2;
	extractor->compute(image1,keypoints1,descriptors1);
	extractor->compute(image2,keypoints2,descriptors2);

	if(descriptors1.type()!=descriptors2.type() || descriptors1.cols!=descriptors2.cols) return 0;

	// 2. Match the two image descriptors
	// Construction of the VMatcher
	//cv::BruteForceMatcher<cv::L2<float>> VMatcher;
	// from image 1 to image 2
	// based on k nearest neighbours (with k=2)
	std::vector<std::vector<cv::DMatch> > matches1;
	matcher->knnMatch(descriptors1,descriptors2,
		matches1, // vector of matches (up to 2 per entry)
		2);        // return 2 nearest neighbours
	// from image 2 to image 1
	// based on k nearest neighbours (with k=2)
	std::vector<std::vector<cv::DMatch> > matches2;
	matcher->knnMatch(descriptors2,descriptors1,
		matches2, // vector of matches (up to 2 per entry)
		2);        // return 2 nearest neighbours

	// 3. Remove matches for which NN ratio is
	// > than threshold
	// clean image 1 -> image 2 matches
	int removed= ratioTest(matches1);
	// clean image 2 -> image 1 matches
	removed= ratioTest(matches2);

	// 4. Remove non-symmetrical matches
	// std::vector<cv::DMatch> symMatches;
	symmetryTest(matches1,matches2,matches);

	/*
	// 5. Validate matches using RANSAC
	cv::Mat fundemental= ransacTest(symMatches,
		keypoints1, keypoints2, matches);
	// return the found fundemental matrix
	return fundemental;
	*/
	
	return matches.size();
}

void VMatcher:: showImage(cv::Mat& image1,
	cv::Mat& image2, // input images
	// output matches and keypoints
	std::vector<cv::DMatch>& matches,
	std::vector<cv::KeyPoint>& keypoints1,
	std::vector<cv::KeyPoint>& keypoints2){
	cv::Mat img_matches;
	cv::drawMatches( image1, keypoints1, image2, keypoints2, 
		matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), 
		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS ); 
	if(matches.size() < 4) {imshow("ImageShow",img_matches); return;} // skip this frames

	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;

	for( int i = 0; i < matches.size(); i++ )
	{
		//-- Get the keypoints from the good matches
		obj.push_back( keypoints1[ matches[i].queryIdx ].pt );
		scene.push_back( keypoints2[ matches[i].trainIdx ].pt );
	}
	cv::Mat H = cv::findHomography( obj, scene, CV_RANSAC ,2);

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<cv::Point2f> obj_corners(4);
	obj_corners[0]=cvPoint(0,0);
	obj_corners[1]=cvPoint(image1.cols, 0 );
	obj_corners[2]=cvPoint(image1.cols,image1.rows);
	obj_corners[3]=cvPoint(0,image1.rows);
	std::vector<cv::Point2f> scene_corners(4);

	cv::perspectiveTransform( obj_corners, scene_corners, H);
	for( int i = 0; i < 4; i++ )
	{
		scene_corners[i].x+=image1.cols;
	}
	line( img_matches, scene_corners[0], scene_corners[1], cv::Scalar(0, 255, 0), 2 );
	line( img_matches, scene_corners[1], scene_corners[2], cv::Scalar( 0, 255, 0), 2 );
	line( img_matches, scene_corners[2], scene_corners[3], cv::Scalar( 0, 255, 0), 2 );
	line( img_matches, scene_corners[3], scene_corners[0], cv::Scalar( 0, 255, 0), 2 );
	imshow("ImageShow",img_matches);
}


