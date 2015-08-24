#include <tracker.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class TrackerGrishin : public Tracker
{
public:
   virtual ~TrackerGrishin() {}

   virtual bool init( const cv::Mat& frame, const cv::Rect& initial_position );
   virtual bool track( const cv::Mat& frame, cv::Rect& new_position );

private:
   cv::Rect position_;
   cv::Mat prevFrame_;

   bool filterAnchors(std::vector<cv::Point2f> &anchors, std::vector<cv::Point2f> &next_anchors, std::vector<uchar> &status, std::vector<float> &errors);
   bool restoreBoundingBox(std::vector<cv::Point2f> &anchors, std::vector<cv::Point2f> &next_anchors, cv::Rect &new_position);
   float get_median(std::vector<float> &v);   
   bool computeMedianShift(std::vector<cv::Point2f> &anchors, std::vector<cv::Point2f> &next_anchors, float &dx, float &dy);
   bool computeScaleFactor(std::vector<cv::Point2f> &corners, std::vector<cv::Point2f> &nextCorners, float &scale);
   bool computePointDistances(std::vector<cv::Point2f> &corners, std::vector<float> &dist);
   bool computeDistScales(std::vector<float> &dist, std::vector<float> &nextDist, std::vector<float> &distScales);
   void drawDetections(const vector<cv::Point2f>& detections, const Scalar& color, Mat& image);
};

void drawDetections(const vector<cv::Point2f>& detections,
                    const Scalar& color,
                    Mat& image)
{
    for (size_t i = 0; i < detections.size(); ++i)
    {
        circle(image, detections[i], 5, color, 1, 8, 0);
    }
}

cv::Ptr<Tracker> createTrackerGrishin()
{
   return cv::Ptr<Tracker>(new TrackerGrishin());
}

float TrackerGrishin::get_median(std::vector<float> &v)
{
	std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

bool TrackerGrishin::filterAnchors(std::vector<cv::Point2f> &anchors, std::vector<cv::Point2f> &next_anchors, std::vector<uchar> &status, std::vector<float> &errors)
{
	for (int i = status.size() - 1; i >= 0; i--)
    {
        if (!status[i])
        {
            status.erase(status.begin() + i);
            anchors.erase(anchors.begin() + i);
			next_anchors.erase(next_anchors.begin() + i);
			next_anchors.erase(next_anchors.begin() + i);
            errors.erase(errors.begin() + i);
        }
    }
    if (anchors.empty())
    {
        return false;
    }

	
	std::vector<float> copy_errors;
	std::copy(errors.begin(), errors.end(), copy_errors.begin());

	float medianError = get_median(copy_errors);

	for (int i = errors.size() - 1; i >= 0; i--)
    {
        if (errors[i] > medianError)
        {
            errors.erase(errors.begin() + i);
            anchors.erase(anchors.begin() + i);
            next_anchors.erase(next_anchors.begin() + i);
            status.erase(status.begin() + i);
        }
    }
    if (anchors.empty())
    {
        return false;
    }

    return true;
}

bool TrackerGrishin::computeMedianShift(std::vector<cv::Point2f> &anchors, std::vector<cv::Point2f> &next_anchors, float &dx, float &dy)
{
    std::vector<float> shiftOx, shiftOy;
    for (int i = 0; i < anchors.size(); i++)
    {
        shiftOx.push_back(next_anchors[i].x - anchors[i].x);
        shiftOy.push_back(next_anchors[i].y - anchors[i].y);
    }
    dx = get_median(shiftOx);
    dy = get_median(shiftOy);
    return true;
}

bool TrackerGrishin::computePointDistances(std::vector<cv::Point2f> &corners, std::vector<float> &dist)
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

bool TrackerGrishin::computeDistScales(std::vector<float> &dist, std::vector<float> &nextDist, std::vector<float> &distScales)
{
    distScales.clear();
    for (int i = 0; i < dist.size(); i++)
    {
        distScales.push_back(nextDist[i] / dist[i]);
    }
    return true;
}

bool TrackerGrishin::computeScaleFactor(std::vector<cv::Point2f> &anchors, std::vector<cv::Point2f> &next_anchors, float &scale)
{
    if (anchors.size() <= 1 || next_anchors.size() <= 1)
    {
        return false;
    }
    std::vector<float> dist, nextDist, distScales;
    computePointDistances(anchors, dist);
    computePointDistances(next_anchors, nextDist);
    computeDistScales(dist, nextDist, distScales);
    scale = get_median(distScales);
    return true;
}

bool TrackerGrishin::restoreBoundingBox(std::vector<cv::Point2f> &anchors, std::vector<cv::Point2f> &next_anchors, cv::Rect &new_position)
{
	float dx, dy;
    computeMedianShift(anchors, next_anchors, dx, dy);
    float ddx = position_.x + dx, ddy = position_.y + dy;
    if (ddx >= prevFrame_.size().width || ddy >= prevFrame_.size().height)
    {
        return false;
    }
    new_position.x = (ddx < 0) ? 0 : ddx;
    new_position.y = (ddy < 0) ? 0 : ddy;

    float scale;
    if (!computeScaleFactor(anchors, next_anchors, scale))
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

bool TrackerGrishin::init( const cv::Mat& frame, const cv::Rect& initial_position )
{
	position_ = initial_position;
	cv::cvtColor(frame, prevFrame_, CV_BGR2GRAY);
	return true;
}

bool TrackerGrishin::track( const cv::Mat& frame, cv::Rect& new_position )
{
	cv::Mat obj = prevFrame_(position_);
	std::vector<cv::Point2f> anchors;

	cv::goodFeaturesToTrack(obj, anchors, 1000, 0.01, 5); 

	if (anchors.empty())
	{
	   std::cout << "Tracked object is lost." << std::endl;
		return false;
	}
    for (int i = 0; i < anchors.size(); i++)
    {
		anchors[i].x += position_.x;
		anchors[i].y += position_.y;
    }	

	std::vector<cv::Point2f> next_anchors;
	std::vector<uchar> status;
	std::vector<float> errors;

	cv::Mat frameGray;
    cv::cvtColor(frame, frameGray, CV_BGR2GRAY);

    cv::calcOpticalFlowPyrLK(prevFrame_, frameGray, 
        anchors, next_anchors, status, errors);


    if (!filterAnchors(anchors, next_anchors, status, errors))
    {
        std::cout << "There are no feature points for tracking." << std::endl;
        return false;
    }

	//drawDetections(anchors, Scalar(255, 0, 0), frame);

	if (!restoreBoundingBox(anchors, next_anchors, new_position))
    {
        std::cout << "There are no enough number of feature points " <<
            "to restore bounding box." << std::endl;
        return false;
    }

	new_position = position_;
	prevFrame_ = frameGray;

	return true;
}
