#include "image_vectorization.h" // Include the new header
#include <iostream>

// Converte uma imagem em um vetor de características HOG normalizado (float, 1xN)
ImageVectorData vectorization(const cv::Mat &imgBgr, const std::string& imageName)
{
    // Pré-processamento: cinza + redimensiona para tamanho fixo (HOG precisa de tamanho consistente)
    cv::Mat imgGray;
    cv::cvtColor(imgBgr, imgGray, cv::COLOR_BGR2GRAY);
    cv::Mat imgResized;
    cv::resize(imgGray, imgResized, cv::Size(128, 128)); // ajuste se quiser

    // HOG padrão (você pode tunar os parâmetros)
    cv::HOGDescriptor hog(
        cv::Size(128, 128), // winSize
        cv::Size(16, 16),   // blockSize
        cv::Size(8, 8),     // blockStride
        cv::Size(8, 8),     // cellSize
        9                   // nbins
    );

    std::vector<float> vectorDesc;
    hog.compute(imgResized, vectorDesc);

    // passa para Mat linha e normaliza (L2)
    cv::Mat vec(vectorDesc, true); // coluna Nx1
    vec = vec.reshape(1, 1);       // 1xN
    cv::normalize(vec, vec, 1.0, 0.0, cv::NORM_L2);

    ImageVectorData data;
    data.vector = vec;
    data.name = imageName;
    return data;
}

// Similaridade do cosseno entre dois vetores 1xN
double cosineSimilarity(const cv::Mat &a, const cv::Mat &b)
{
    try
    {
        CV_Assert(a.rows == 1 && b.rows == 1 && a.cols == b.cols && a.type() == CV_32F && b.type() == CV_32F);

        double dot = a.dot(b);
        double na = cv::norm(a);
        double nb = cv::norm(b);
        return (na > 0 && nb > 0) ? dot / (na * nb) : 0.0;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 0.0;
    }
}
