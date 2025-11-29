#include "mtree.h"
#include "my_header.h"
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace fs = std::filesystem;
using namespace std;

// Funcao auxiliar para verificar se um arquivo e uma imagem
static bool isImageFile(const fs::path& path) {
    if (!fs::is_regular_file(path)) return false;
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
    return (ext == ".png" || ext == ".jpg" || ext == ".jpeg");
}

// Declaracao da funcao logExecutionTime (ja existe em my_header.cpp)
void logExecutionTime(const std::string& logFileName, const std::string& imageName, size_t listSize, double durationNs);

void processWithMTree(const std::string& folderBD, const std::string& folderComparar) {
    // Criar M-Tree com distancia derivada do cosseno
    MTree mtree(MTree::cosineDistance);
    mtree.buildFromDirectory(folderBD);

    // Imprimir estatisticas da arvore
    mtree.printStats();

    for (const auto &entry : fs::directory_iterator(folderComparar)) {
        if (!isImageFile(entry.path())) {
            continue;
        }

        const std::string queryImagePath = entry.path().string();

        // Mede o tempo da consulta
        auto start = std::chrono::high_resolution_clock::now();

        // Buscar todos os vizinhos (k=-1)
        auto resultados = mtree.queryFromImage(queryImagePath, -1);

        // Para o cronometro
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::nano> duration = end - start;

        // Salva o log do tempo de execucao
        logExecutionTime("log/mtree_log.csv", queryImagePath, resultados.size(), duration.count());

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
        cout << "\nTempo de busca: " << duration.count() << " ns\n" << endl;
        std::cout << "--------------------------------------------------------\n\n";
    }
}
