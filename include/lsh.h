#ifndef LSH_H
#define LSH_H

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include "image_vectorization.h"

class LSHIndex {
public:
    struct Result {
        std::string image_name;
        double similarity;
    };

    LSHIndex(
        int dim,
        int num_tables = 8,
        int num_planes = 18,
        unsigned int seed = 42
    );

    void add(const ImageVectorData& data);
    void addMultiple(const std::vector<ImageVectorData>& data_vector);
    std::vector<Result> query(const cv::Mat& q, int top_k = 5, int max_candidates = 0);

private:
    int dim_;
    int num_tables_;
    int num_planes_;

    std::mt19937 rng_;
    std::normal_distribution<double> dist_;

    std::vector<std::vector<cv::Mat>> hyperplanes_;

    struct StoredData {
        std::string name;
        cv::Mat vector;
    };
    std::vector<std::unordered_map<std::string, std::vector<StoredData>>> tables_;

    void generateHyperplanes();
    std::vector<std::string> hashVector(const cv::Mat& v);
};

#endif // LSH_H
