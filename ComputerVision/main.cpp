
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <stack>
#include "Graph.h"
#include <opencv2\ml.hpp>
#include <stack>
#include <vector>



int main(int argc, const char** argv)
{
   
    //Declarations 
    cv::Mat frame, gray, frameDelta, thresh, firstFrame, histImage, b_hist;
    int histSize = 256;

    //set the ranges
    float range[] = { 15, 256 };
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
 
    
    
    _histogram::Graph d;
    //------------------------------//
    std::vector<std::vector<cv::Point> > cnts;
    std::vector<int> peaks;
    cv::VideoCapture camera(0); //open camera
    static int resetcam = 0;

    //set the video size to 512x288 to process faster
    camera.set(3, 500);
    camera.set(4, 280);
    /// sleep. Wait for camera set
    std::this_thread::sleep_for(std::chrono::milliseconds(3));


    camera.read(frame);

    //convert to grayscale and set the first frame

    cvtColor(frame, firstFrame, cv::COLOR_BGR2GRAY);
    GaussianBlur(firstFrame, firstFrame, cv::Size(21, 21), 0);


    while (camera.read(frame)) {
        if (resetcam == 100 || resetcam == 0) {
            //Resets the frame after 10 seconds
            camera.read(frame);
            cvtColor(frame, firstFrame, cv::COLOR_BGR2GRAY);
            GaussianBlur(firstFrame, firstFrame, cv::Size(21, 21), 0);
            resetcam = 1;
        }

        //convert to grayscale
        cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, cv::Size(21, 21), 0);
       
        //compute difference between first frame and current frame
        absdiff(firstFrame, gray, frameDelta); //Input needs to be in grayscale
        threshold(frameDelta, thresh, 15, 256, cv::THRESH_BINARY);
        cv::erode(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 1); //Noise Reduction
        dilate(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 1); //Noise reduction
        findContours(thresh, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); //Finding contours
       
       // d.histogramcalculation(gray, histImage);
        cv::calcHist(&frameDelta, 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
        //d.findHistPeaks(histImage, peaks, 0);
        peaks = d.getLocalMaximum(b_hist);
        for (int i = 0; i < cnts.size(); i++) {
            if (contourArea(cnts[i]) < 300) {
                
                continue;
            }
           
            
            putText(frame, "Motion Detected", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 0), 2);
        }
       
        imshow("Camera", frame);
        imshow("Thresh", thresh);
      
       
       
        if (cv::waitKey(1) == 27) {
          
            //exit if ESC is pressed
            break;
        }
        resetcam++;
    }
    
    return 0;
}