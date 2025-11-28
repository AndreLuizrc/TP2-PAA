#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "image_vectorization.h"
#include "my_header.h"

namespace fs = std::filesystem;
using namespace std;

int main1() {
    string folderBD = "imgs/BD";
    string folderComparar = "imgs/comparar";

    vector<ImageVectorData> bdImagens = loadImages(folderBD, "BD");
    vector<ImageVectorData> vCompararImagens = loadImages(folderComparar, "comparar");

    if (bdImagens.empty() || vCompararImagens.empty()) {
        cerr << "Erro: BD ou comparar estao vazios." << endl;
        return 1;
    }

    for (const auto& imgComparar : vCompararImagens) {
        processAndPrintSimilarities(imgComparar, bdImagens);
    }

    printTable(vCompararImagens, bdImagens, vCompararImagens.size());

    return 0;
}
