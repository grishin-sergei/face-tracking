#include <tracker.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include <stdio.h>
#include "opencv2/core/core.hpp"
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

//#include "tbb/task_scheduler_init.h"

using namespace cv;
using namespace std;

class TrackerKustikova : public Tracker
{
 public:
    virtual ~TrackerKustikova() {}

    virtual bool init( const cv::Mat& frame, const cv::Rect& initial_position );
    virtual bool track( const cv::Mat& frame, cv::Rect& new_position );

 private:
    cv::Rect position_;
    cv::Mat prevFrame_;

    bool filterCorners(std::vector<cv::Point2f> &corners, 
        std::vector<cv::Point2f> &nextCorners, std::vector<uchar> &status,
        std::vector<float> &errors);

	bool filterRANSAC(std::vector<cv::Point2f> &corners, 
        std::vector<cv::Point2f> &nextCorners, cv::Mat newFrame_);

    bool restoreBoundingBox(std::vector<cv::Point2f> &corners, 
        std::vector<cv::Point2f> &nextCorners, cv::Rect &new_position);

    float median(std::vector<float> &v);

    bool computeMedianShift(std::vector<cv::Point2f> &corners, 
        std::vector<cv::Point2f> &nextCorners, float &dx, float &dy);

    bool computeScaleFactor(std::vector<cv::Point2f> &corners, 
        std::vector<cv::Point2f> &nextCorners, float &scale);

    bool computePointDistances(std::vector<cv::Point2f> &corners,
        std::vector<float> &dist);

    bool computeDistScales(std::vector<float> &dist, 
        std::vector<float> &nextDist, std::vector<float> &distScales);
};

void drawDetections(const cv::vector<cv::Point2f>& detections, const cv::Scalar& color, cv::Mat image)
{
    for (size_t i = 0; i < detections.size(); ++i)
    {
        circle(image, detections[i], 3, color, 1, 8, 0);
    }
}

bool TrackerKustikova::init( const cv::Mat& frame, const cv::Rect& initial_position )
{
    position_ = initial_position;
    cv::cvtColor(frame, prevFrame_, CV_BGR2GRAY);

    return true;
}

float TrackerKustikova::median(std::vector<float> &v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

bool TrackerKustikova::filterCorners(std::vector<cv::Point2f> &corners, 
        std::vector<cv::Point2f> &nextCorners, std::vector<uchar> &status,
        std::vector<float> &errors)
{
    for (int i = status.size() - 1; i >= 0; i--)
    {
        if (!status[i])
        {
            status.erase(status.begin() + i);
            corners.erase(corners.begin() + i);
            nextCorners.erase(nextCorners.begin() + i);
            errors.erase(errors.begin() + i);
        }
    }
    if (corners.empty())
    {
        return false;
    }
    std::vector<float> errorsCopy(errors.size());
    std::copy(errors.begin(), errors.end(), errorsCopy.begin());
    float medianError = median(errorsCopy);

    for (int i = errors.size() - 1; i >= 0; i--)
    {
        if (errors[i] > medianError)
        {
            errors.erase(errors.begin() + i);
            corners.erase(corners.begin() + i);
            nextCorners.erase(nextCorners.begin() + i);
            status.erase(status.begin() + i);
        }
    }
    if (corners.empty())
    {
        return false;
    }

    return true;
}

bool TrackerKustikova::filterRANSAC(std::vector<cv::Point2f> &corners, 
        std::vector<cv::Point2f> &nextCorners, cv::Mat newFrame_)
{
	int ransacReprojThreshold = 3;

	cv::Mat prev_(prevFrame_(position_));
	cv::Mat new_(newFrame_);

	// detecting keypoints
   // SurfFeatureDetector detector;
    vector<KeyPoint> keypoints1;
	for(int i = 0; i < corners.size(); i++)
	{
		keypoints1.push_back(KeyPoint(corners[i].x, corners[i].y, 1));
	}
   // detector.detect(prev_, keypoints1);
    vector<KeyPoint> keypoints2;
	for(int i = 0; i < nextCorners.size(); i++)
	{
		keypoints2.push_back(KeyPoint(nextCorners[i].x, nextCorners[i].y, 1));
	}
   // detector.detect(new_, keypoints2);
	//std::cout << keypoints1.size() << " " << keypoints2.size() << std::endl;

    // computing descriptors
    SurfDescriptorExtractor extractor;
    Mat descriptors1;
    extractor.compute(prev_, keypoints1, descriptors1);
    Mat descriptors2;
    extractor.compute(newFrame_, keypoints2, descriptors2);

    // matching descriptors
    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
	Mat img_matches;
	
    //waitKey(0);
	
	std::cout << matches.size() << std::endl;

	vector<Point2f> points1, points2;
    // fill the arrays with the points
    for (int i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints1[i].pt);
    }
    for (int i = 0; i < matches.size(); i++)
    {
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    Mat H = findHomography(Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold);

    Mat points1Projected;
    perspectiveTransform(Mat(points1), points1Projected, H);
	


	//int minHessian = 15;

	//cv::FastFeatureDetector detector( minHessian );

	//int ransacDistance = 3;

	//cv::Mat prev_(prevFrame_);
	//cv::Mat new_(newFrame_);   

	//std::vector<cv::KeyPoint> keypoints_object;
	//std::vector<cv::KeyPoint> keypoints_scene;

 //   detector.detect( prev_, keypoints_object );
 //   detector.detect( new_, keypoints_scene );

	//cv::SurfDescriptorExtractor extractor;

	//cv::Mat descriptors_object, descriptors_scene;

 //   extractor.compute( prev_, keypoints_object, descriptors_object );
 //   extractor.compute( new_, keypoints_scene, descriptors_scene );

	//cv::BFMatcher matcher;

 //   std::vector<cv::DMatch > matches;
 //   matcher.match( prev_, new_, matches);

	////-- Localize the object
 //   std::vector<cv::Point2f> obj;
 //   std::vector<cv::Point2f> scene;
	//
 //   for( int i = 0; i < matches.size(); i++ )
	//{
	//	//-- Get the keypoints from the good matches
	//	obj.push_back( corners[ matches[i].queryIdx ] );
	//	scene.push_back( nextCorners[ matches[i].trainIdx ] );
 //   }

	//cv::Mat H = cv::findHomography( obj, scene, CV_RANSAC, ransacDistance );

	//std::vector<cv::Point2f> obj_corners;
	//std::vector<cv::Point2f> scene_corners;

	//cv::perspectiveTransform( obj_corners, scene_corners, H);

	//cv::vector<cv::DMatch> inliner;

	std::vector<cv::Point2f> new_corners, new_nextCorners;

	for(int i = 0; i < matches.size(); i++)
	{
		Point2f p1 = points1Projected.at<Point2f>(matches[i].queryIdx);
        Point2f p2 = keypoints2.at(matches[i].trainIdx).pt;
		if(((p2.x - p1.x) * (p2.x - p1.x) +
			(p2.y - p1.y) * (p2.y - p1.y) <= ransacReprojThreshold * ransacReprojThreshold) )
		{
			//inliner.push_back(matches[i]);
			//&& ((p2.x > position_.x - 10) &&(p2.x < position_.x + 10) && (p2.y > position_.y - 10) &&(p2.y < position_.y + 10))

			new_corners.push_back(p1);
			new_nextCorners.push_back(p2);
		}		
	}
//	drawMatches(prev_, keypoints1, newFrame_, keypoints2, matches, img_matches);
 //   imshow("matches before homography", img_matches);

	corners = new_corners;
	nextCorners = new_nextCorners;

    if (corners.empty())
    {
        return false;
    }

    return true;
}

bool TrackerKustikova::computeMedianShift(std::vector<cv::Point2f> &corners, 
        std::vector<cv::Point2f> &nextCorners, float &dx, float &dy)
{
    std::vector<float> shiftOx, shiftOy;
    for (int i = 0; i < corners.size(); i++)
    {
        shiftOx.push_back(nextCorners[i].x - corners[i].x);
        shiftOy.push_back(nextCorners[i].y - corners[i].y);
    }
    dx = median(shiftOx);
    dy = median(shiftOy);
    return true;
}

bool TrackerKustikova::computePointDistances(std::vector<cv::Point2f> &corners,
        std::vector<float> &dist)
{
    dist.clear();
    for (int i = 0; i < corners.size(); i++)
    {
        for (int j = i + 1; j < corners.size(); j++)
        {
            dist.push_back(cv::norm(corners[i] - corners[j]));
        }
    }
    return true;
}

bool TrackerKustikova::computeDistScales(std::vector<float> &dist, 
        std::vector<float> &nextDist, std::vector<float> &distScales)
{
    distScales.clear();
    for (int i = 0; i < dist.size(); i++)
    {
        distScales.push_back(nextDist[i] / dist[i]);
    }
    return true;
}

bool TrackerKustikova::computeScaleFactor(std::vector<cv::Point2f> &corners,
        std::vector<cv::Point2f> &nextCorners, float &scale)
{
    if (corners.size() <= 1 || nextCorners.size() <= 1)
    {
        return false;
    }
    std::vector<float> dist, nextDist, distScales;
    computePointDistances(corners, dist);
    computePointDistances(nextCorners, nextDist);
    computeDistScales(dist, nextDist, distScales);
    scale = median(distScales);
    return true;
}

bool TrackerKustikova::restoreBoundingBox(std::vector<cv::Point2f> &corners,
        std::vector<cv::Point2f> &nextCorners, cv::Rect &new_position)
{
    float dx, dy;
    computeMedianShift(corners, nextCorners, dx, dy);
    float ddx = position_.x + dx, ddy = position_.y + dy;
    if (ddx >= prevFrame_.size().width || ddy >= prevFrame_.size().height)
    {
        return false;
    }
    new_position.x = (ddx < 0) ? 0 : ddx;
    new_position.y = (ddy < 0) ? 0 : ddy;

    float scale;
    if (!computeScaleFactor(corners, nextCorners, scale))
    {
        return false;
    }
    float width, height;
    width = position_.width * scale;
    height = position_.height * scale;

    float x = ddx + width, y = ddy + height;
    x = (x >= prevFrame_.size().width) ? prevFrame_.size().width - 1 : x;
    y = (y >= prevFrame_.size().height) ? prevFrame_.size().height - 1 : y;
    new_position.width = x - new_position.x;
    new_position.height = y - new_position.y;

    return true;
}

bool TrackerKustikova::track( const cv::Mat& frame, cv::Rect& new_position )
{
    cv::Mat object = prevFrame_(position_);
    std::vector<cv::Point2f> corners;
    const int maxCorners = 100;
    cv::goodFeaturesToTrack(object, corners, maxCorners, 0.01, 5);
    if (corners.empty())
    {
        std::cout << "Tracked object is lost." << std::endl;
        return false;
    }
    for (int i = 0; i < corners.size(); i++)
    {
        corners[i].x += position_.x;
        corners[i].y += position_.y;
    }

    std::vector<cv::Point2f> nextCorners;
	cv::Mat frameGray;
	cv::cvtColor(frame, frameGray, CV_BGR2GRAY);
    std::vector<uchar> status;
    std::vector<float> errors;    
    cv::calcOpticalFlowPyrLK(prevFrame_, frameGray, 
        corners, nextCorners, status, errors);

	/*if (!filterRANSAC(corners, nextCorners, frameGray))
    {
        std::cout << "There are no feature points for tracking." << std::endl;
        return false;
    }*/

    if (!filterCorners(corners, nextCorners, status, errors))
    {
        std::cout << "There are no feature points for tracking." << std::endl;
        return false;
    }

	drawDetections(corners, cv::Scalar(255, 0, 0), frame);

    if (!restoreBoundingBox(corners, nextCorners, new_position))
    {
        std::cout << "There are no enough number of feature points " <<
            "to restore bounding box." << std::endl;
        return false;
    }

    position_ = new_position;
    prevFrame_ = frameGray;
	return true;
}

cv::Ptr<Tracker> createTrackerKustikova()
{
    return cv::Ptr<Tracker>(new TrackerKustikova());
}