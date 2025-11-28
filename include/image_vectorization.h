#ifndef IMAGE_VECTORIZATION_H
#define IMAGE_VECTORIZATION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/** Estrutura do vetor de caracteristicas
 * @param vector {cv::Mat} Vetor de características da imagem
 * @param name {std::string} Nome do arquivo da imagem
*/
struct ImageVectorData {
    cv::Mat vector;
    std::string name;
};

ImageVectorData vectorization(const cv::Mat& imgBgr, const std::string& imageName);

double cosineSimilarity(const cv::Mat& a, const cv::Mat& b);

#endif // IMAGE_VECTORIZATION_H
