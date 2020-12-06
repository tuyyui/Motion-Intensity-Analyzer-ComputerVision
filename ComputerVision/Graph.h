#ifndef GRAPH_H
#define GRAPH_H
#include <cmath>

#include <iostream>
#include <fstream>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2\ml.hpp>

using namespace cv;
#define DEBUG_HISTOGRAM 1
#define EXIT 0
#define START_TIMEH std::chrono::high_resolution_clock::now()
struct Length
{
    int pos1;
    int pos2;
    int size()
    {
        return pos2 - pos1 + 1;
    }
};

struct PeakInfo
{
    int pos;
    int left_size;
    int right_size;
    float value;
};

namespace _histogram {

    class Graph {

    public:
        /// 
        /// Draw red-lines on each peak on debug. Reading the peaks into a file
        ///     
        int drawPeaks(Mat& histImage, std::vector<int>& peaks, int hist_size = 256, Scalar color = Scalar(0, 0, 255));
       
        ///
        /// Drawing the histogram
        ///
        /// 
        Mat drawHistogram(Mat& hist, int hist_h = 400, int hist_w = 1024, int hist_size = 256, Scalar color = Scalar(255, 255, 255), int type = 2);
      
        /// 
        /// Get information of highest peak
        ///
        PeakInfo peakInfo(int pos, int left_size, int right_size, float value);

        /// 
        ///  Find the largest peak of the curve
        /// 
        std::vector<PeakInfo> findPeaks(InputArray _src, int window_size);

        /// 
        /// Get the local maximum of the bell curve
        ///   
        std::vector<int> getLocalMaximum(InputArray _src, int smooth_size = 9, int neighbor_size = 3, float peak_per = 0.5); //if you play with the peak_per attribute value, you can increase/decrease the number of peaks found

    };
  
};
#endif