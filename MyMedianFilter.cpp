//
//  main.cpp
//  OpenCV
//
//  Created by 이동환 on 2022/04/09.
//

#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

void Gauss1(InputArray input, OutputArray output, float sigma) {
    const Mat& img = input.getMat();
    output.create(img.size(), img.type());
    Mat dst = output.getMat();
    int windowSize = int(ceil(sigma*2)+1);
    for (int y=0; y<img.rows; y++) {
        for (int x=0; x<img.cols; x++) {
            
            float wSum = 0;
            float vSum = 0;
            for (int t = -windowSize; t<=windowSize; t++) {
                for (int s = -windowSize; s<=windowSize; s++) {
                    float g = exp(-(s*s + t*t)/(sigma*sigma));
                    wSum += g;
                    vSum += g*img.at<float>(min(img.rows-1, max(0,y+t)), min(img.cols-1, max(0,x+s)));
                }
            }
            dst.at<float> (y,x) = vSum / wSum;
        }
    }
    
}

void myMedianFilter(InputArray input, OutputArray output, int windowSize) {
    const Mat& img = input.getMat();
    output.create(img.size(), img.type());
    Mat dst = output.getMat();
    int range = (windowSize-1)/2;
    for (int y=0; y<img.rows; y++) {
        for (int x=0; x<img.cols; x++) {
            
            std::vector <uchar> pixels;
            for (int t = -range; t<=range; t++) {
                for (int s = -range; s<=range; s++) {
                    uchar eachPixel = img.at<uchar>(min(img.rows-1, max(0,y+t)), min(img.cols-1, max(0,x+s)));
                    pixels.push_back(eachPixel);
                }
            }
            std::sort(pixels.begin(), pixels.end());
            uchar median = pixels[windowSize/2];
            dst.at<uchar>(y,x) = median;
        }
    }
}

int main() {
    
    Mat img = imread("/Users/donghwan/Desktop/3-1/영상처리/MedianFilterInput.png", 0);
    Mat output;
    Mat standard;
    myMedianFilter(img, output, 3);
    medianBlur(img, standard, 5);
    
    imshow("MedianFilterInput", img);
    imshow("MedianFilterOutput", output);
    imshow("StandardMedianFilter", standard);
    waitKey();
    return 0;
}
