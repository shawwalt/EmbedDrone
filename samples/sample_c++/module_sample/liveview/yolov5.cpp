//
// Created by shawalt on 2025/5/7.
//
#include "yolov5.hpp"

std::vector<std::string> Yolov5::get_class_list() {
    return class_list;
}

void Yolov5::load_class_list(std::string path) {
    class_list.clear();
    std::vector<std::string> classes;
    std::ifstream ifs(path);
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
}

void Yolov5::load_net(std::string path) {
    USER_LOG_INFO("net loaded");
    auto result = cv::dnn::readNet(path);
    if (is_cuda) {
        std::cout << "Attempting to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    } else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

cv::Mat Yolov5::format_yolov5(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

Yolov5::Yolov5(bool use_cuda) : is_cuda(use_cuda) {
}

void Yolov5::detect(cv::Mat &image, std::vector<Detection> &output) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1./255.,
                           cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
                           cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;

    auto start = std::chrono::high_resolution_clock::now();
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;

    std::cout << "Forward pass execution time: " << duration.count() << " seconds" << std::endl;

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float *classes_scores = data + 5;
            cv::Mat scores(1, class_list.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
//            printf("confidence: %f", confidence);
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
//            printf(" max score: %f\n", max_class_score);
            if (max_class_score > SCORE_THRESHOLD) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

void Yolov5::process_video(const std::string& video_path) {
    cv::Mat frame;
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) {
        std::cerr << "Error opening video file\n";
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    while (true) {
        capture.read(frame);
        if (frame.empty()) {
            std::cout << "End of stream\n";
            break;
        }

        std::vector<Detection> output;
        detect(frame, output);

        frame_count++;
        total_frames++;

        for (const auto& detection : output) {
            auto box = detection.box;
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];
            cv::rectangle(frame, box, color, 3);
            cv::rectangle(frame, cv::Point(box.x, box.y - 20),
                          cv::Point(box.x + box.width, box.y),
                          color, cv::FILLED);
            cv::putText(frame, class_list[classId].c_str(),
                        cv::Point(box.x, box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0));
        }

        if (frame_count >= 30) {
            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 /
                  std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        if (fps > 0) {
            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            cv::putText(frame, fps_label.str().c_str(),
                        cv::Point(10, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 1,
                        cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("output", frame);

        if (cv::waitKey(1) != -1) {
            capture.release();
            std::cout << "finished by user\n";
            break;
        }
    }

    std::cout << "Total frames: " << total_frames << "\n";
}
