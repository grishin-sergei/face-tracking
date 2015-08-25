#include "tracker.hpp"

cv::Ptr<Tracker> createFaceTracker();
// TODO: Declare your implementation here
 //cv::Ptr<Tracker> createTrackerGrishin();
cv::Ptr<Tracker> createTrackerKustikova();

cv::Ptr<Tracker> createTracker(const std::string &impl_name)
{
    if (impl_name == "FaceTracker")
       // return createFaceTracker();
	return createTrackerKustikova();
    // TODO: Add case for your implementation
    // else if (impl_name == "grishin")
      //  return createTrackerGrishin();
	else if (impl_name == "Kustikova")
        return createTrackerKustikova();

    return 0;
}
