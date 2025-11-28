#include "my_header.h"
#include <bits/stdc++.h>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime> 
#include "arvore.h"

namespace fs = std::filesystem;
using namespace std;

void print_message() {
    std::cout << "This is a message from the header file!" << std::endl;
}

// Carregar imagens
vector<ImageVectorData> loadImages(const string& folderPath, const string& tipo){

    vector<ImageVectorData> imagens;

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        string path = entry.path().string();
        string filename = entry.path().filename().string();
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            cerr << "Erro ao carregar (" << tipo << "): " << filename << endl;
            continue;
        }
        imagens.push_back(vectorization(img, filename));
    }

    return imagens;
}

// Print cabecalho tabela
void printHeader(const vector<ImageVectorData>& compararImagens){
    // Construir tabela
    cout << "Tabela de Similaridade:" << endl;
    cout << "_____________________________________________________________________" << endl;

    // Cabecalho
    cout << "|" << setw(20) << left << "Imagem BD";
    for (auto& cmp : compararImagens) {
        cout << "|" << setw(15) << left << cmp.name;
    }
    cout << "|" << endl;
}

// Print linhas tabela
void printRows(const vector<ImageVectorData>& bdImagens, 
               const vector<ImageVectorData>& compararImagens){
    // Linha separadora
    cout << "|" << string(20, '-') ;
    for (size_t i = 0; i < compararImagens.size(); i++) {
        cout << "|" << string(15, '-');
    }
    cout << "|" << endl;

    // Linhas da tabela
    for (auto& bd : bdImagens) {
        cout << "|" << setw(20) << left << bd.name;
        for (auto& cmp : compararImagens) {
            double sim = cosineSimilarity(bd.vector, cmp.vector);
            cout << "|" << setw(15) << fixed << setprecision(4) << sim;
        }
        cout << "|" << endl;
    }
}

// print rodape tabela
void printFooter(size_t compararSize){
    // Rodapé
    cout << "|" << string(20, '_');
    for (size_t i = 0; i < compararSize; i++) {
        cout << "|" << string(15, '_');
    }
    cout << "|" << endl;
}

void printTable(const vector<ImageVectorData>& compararImagens,
                const vector<ImageVectorData>& bdImagens,
                size_t compararSize) {
    printHeader(compararImagens);
    printRows(bdImagens, compararImagens);
    printFooter(compararSize);
}


// -----------------------------------------------------------------
// LISTA
// -----------------------------------------------------------------

/** Salva uma entrada de log em um arquivo CSV.
 * @param logFileName O nome do arquivo de log
 * @param imageName O nome da imagem que foi comparada.
 * @param listSize O número de imagens na base de dados com a qual foi comparada.
 * @param durationNs O tempo de ordenação em nanossegundos.
 */
void logExecutionTime(const std::string& logFileName, const std::string& imageName, size_t listSize, double durationNs) {
    std::ofstream logFile(logFileName, std::ios::app);

    if (!logFile.is_open()) {
        std::cerr << "Erro: Nao foi possivel abrir o arquivo de log: " << logFileName << std::endl;
        return;
    }

    logFile.seekp(0, std::ios::end);
    if (logFile.tellp() == 0) {
        logFile << "Timestamp,ImagemComparada,TamanhoDaLista,DuracaoOrdenacao_ns\n";
    }

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::tm buf;
    #ifdef _WIN32
    // Versão para  Windows
        localtime_s(&buf, &in_time_t);
    #else
        // Versão para Linux, macOS e outros sistemas POSIX
        localtime_r(&in_time_t, &buf);
    #endif
    
    logFile << std::put_time(&buf, "%Y-%m-%d %H:%M:%S") << ","
            << imageName << ","
            << listSize << ","
            << durationNs << "\n";
}

/* QUICK SORT - https://www.geeksforgeeks.org/cpp/cpp-program-for-quicksort/ - Adaptado */
int partition(vector<SimilarityResult> &vec, int low, int high) {

    // Selecting last element as the pivot
    double pivotSimilarity = vec[high].similarity;

    // Index of elemment just before the last element
    // It is used for swapping
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {

        // If current element is smaller than or
        // equal to pivot
        if (vec[j].similarity > pivotSimilarity) {
            i++;
            swap(vec[i], vec[j]);
        }
    }

    // Put pivot to its position
    swap(vec[i + 1], vec[high]);

    // Return the point of partition
    return (i + 1);
}

void quickSort(vector<SimilarityResult> &vec, int low, int high) {

    // Base case: This part will be executed till the starting
    // index low is lesser than the ending index high
    if (low < high) {

        // pi is Partitioning Index, arr[p] is now at
        // right place
        int pi = partition(vec, low, high);

        // Separately sort elements before and after the
        // Partition Index pi
        quickSort(vec, low, pi - 1);
        quickSort(vec, pi + 1, high);
    }
}



/** Processar e imprimir as similaridades
 * @param imgComparar {const ImageVectorData&} 
 * @param bdImagens {const std::vector<ImageVectorData>&} 
 */
void processAndPrintSimilarities(const ImageVectorData& imgComparar, const std::vector<ImageVectorData>& bdImagens) {
    cout << "--------------------------------------------------------" << endl;
    cout << "Comparando imagem: " << imgComparar.name << endl;
    cout << "--------------------------------------------------------" << endl;

    vector<SimilarityResult> similarityList;

    // Calcula a similaridade com todas as imagens do BD
    for (const auto& bdImg : bdImagens) {
        double sim = cosineSimilarity(imgComparar.vector, bdImg.vector);
        similarityList.push_back({bdImg, sim});
    }

    // Mede o tempo de ordenação
    auto start = chrono::high_resolution_clock::now();

    // Ordena a lista de similaridades em ordem decrescente
    if (!similarityList.empty()) {
        quickSort(similarityList, 0, similarityList.size() - 1);
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, nano> duration = end - start;

    // Salva o resultado da execução no arquivo de log
    logExecutionTime("log/list_log.csv", imgComparar.name, similarityList.size(), duration.count());

    // Imprime a lista ordenada
    cout << "Lista de similaridade ordenada:" << endl;
    cout << setw(20) << left << "Imagem BD" << "| " << "Similaridade" << endl;
    cout << string(20, '-') << "|-" << string(15, '-') << endl;

    for (const auto& result : similarityList) {
        cout << setw(20) << left << result.bdImage.name << "| "
             << fixed << setprecision(4) << result.similarity << endl;
    }

    cout << "\nTempo de ordenacao: " << duration.count() << " ns\n" << endl;
}


// -----------------------------------------------------------------
// KD Tree
// -----------------------------------------------------------------

// Função auxiliar para verificar se um arquivo é uma imagem
static bool isImageFile(const fs::path& path) {
    if (!fs::is_regular_file(path)) return false;
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
    return (ext == ".png" || ext == ".jpg" || ext == ".jpeg");
}

void processWithKDTree(const std::string& folderBD, const std::string& folderComparar) {
    KDTree2D kd;
    kd.buildIndex(folderBD);

    for (const auto &entry : fs::directory_iterator(folderComparar)) {
        if (!isImageFile(entry.path())) {
            continue;
        }

        const std::string queryImagePath = entry.path().string();

        // Mede o tempo da consulta
        auto start = std::chrono::high_resolution_clock::now();

        auto resultados = kd.queryAllFromImage(queryImagePath, -1);
        
        // Para o cronometro
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::nano> duration = end - start;
        
        // Salva o log do tempo de execução
        logExecutionTime("log/kdtree_log.csv", queryImagePath, resultados.size(), duration.count());

        if (resultados.empty()) {
            std::cerr << "Nao foi possivel processar a imagem ou nenhum resultado foi encontrado para: " << queryImagePath << "\n";
            continue;
        }

        std::cout << "--------------------------------------------------------" << endl;
        std::cout << "Comparando imagem: " << fs::path(queryImagePath).filename().string() << std::endl;
        std::cout << "--------------------------------------------------------" << endl;
        for (const auto &[nome, score] : resultados) {
            std::cout << "  " << std::setw(25) << std::left << nome
                      << "-> Similaridade: " << std::fixed << std::setprecision(4) << score << "\n";
        }
        cout << "\nTempo de ordenacao: " << duration.count() << " ns\n" << endl;
        std::cout << "--------------------------------------------------------\n\n";
    }
}