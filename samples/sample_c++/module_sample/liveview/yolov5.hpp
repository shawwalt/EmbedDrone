//
// Created by shawalt on 2025/5/7.
//

#ifndef ENTRY_YOLOV5_HPP
#define ENTRY_YOLOV5_HPP

#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include <stdio.h>
#include <dji_logger.h>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

class Yolov5 {
private:
    std::vector<std::string> class_list;
    cv::dnn::Net net;
    bool is_cuda;
    const std::vector<cv::Scalar> colors = {
            cv::Scalar(255, 255, 0),
            cv::Scalar(0, 255, 0),
            cv::Scalar(0, 255, 255),
            cv::Scalar(255, 0, 0)
    };

    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float SCORE_THRESHOLD = 0.2;
    const float NMS_THRESHOLD = 0.4;
    const float CONFIDENCE_THRESHOLD = 0.4;

    cv::Mat format_yolov5(const cv::Mat &source);

public:
    Yolov5(bool use_cuda = true);
    void load_net(std::string path);
    void load_class_list(std::string path);
    void detect(cv::Mat &image, std::vector<Detection> &output);
    void process_video(const std::string& video_path);
    std::vector<std::string> get_class_list();
};

#endif //ENTRY_YOLOV5_HPP
