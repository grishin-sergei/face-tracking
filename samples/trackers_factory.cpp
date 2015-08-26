#include "tracker.hpp"
#include <iostream>

cv::Ptr<Tracker> createFaceTracker();
// TODO: Declare your implementation here
 //cv::Ptr<Tracker> createTrackerGrishin();
cv::Ptr<Tracker> createTrackerForProject();

cv::Ptr<Tracker> createTracker(const std::string &impl_name)
{
    if (impl_name == "ForProject")
        return createTrackerForProject();
	else if(impl_name == "FaceTracker")
		return createFaceTracker();
		

    return 0;
}
