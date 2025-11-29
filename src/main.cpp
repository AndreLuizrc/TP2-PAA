#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>
#include "arvore.h"
#include <opencv2/opencv.hpp>
#include "image_vectorization.h"
#include "my_header.h"
#include "lsh.cpp" // Incluir a implementação da classe LSHIndex

namespace fs = std::filesystem;

// Função para processar e imprimir resultados LSH
void processWithLSH(const std::string& folderBD, const std::string& folderComparar) {
    std::cout << "--------------------------------------------------------\n\n";
    std::cout << "  LSH\n\n";
    std::cout << "--------------------------------------------------------\n\n";

    LSHIndex lsh(512); // Dimensão 512, ajuste se necessário

    std::vector<ImageVectorData> bdImagensLSH = loadImages(folderBD, "BD");
    if (bdImagensLSH.empty()) {
        std::cerr << "Erro: BD esta vazio para LSH." << std::endl;
        return;
    }
    lsh.addMultiple(bdImagensLSH);

    std::vector<ImageVectorData> vCompararImagensLSH = loadImages(folderComparar, "comparar");
    if (vCompararImagensLSH.empty()) {
        std::cerr << "Erro: comparar esta vazio para LSH." << std::endl;
        return;
    }

    for (const auto& imgComparar : vCompararImagensLSH) {
        std::vector<LSHIndex::Result> results = lsh.query(imgComparar.vector, 5); // top 5
        std::cout << "Imagens similares para " << imgComparar.name << " (LSH):\n";
        for (const auto& result : results) {
            std::cout << "  - " << result.image_name << " (Similaridade: " << std::fixed << std::setprecision(4) << result.similarity << ")\n";
        }
        std::cout << "\n";
    }
}

int main()
{
  const std::string folderBD = "imgs/BD";
  const std::string folderComparar = "imgs/comparar";

  processWithLSH(folderBD, folderComparar);

  /* LISTA */
  std::cout << "--------------------------------------------------------\n\n";
  std::cout << "  LIST\n\n";
  std::cout << "--------------------------------------------------------\n\n";
  std::vector<ImageVectorData> bdImagens = loadImages(folderBD, "BD");
  std::vector<ImageVectorData> vCompararImagens = loadImages(folderComparar, "comparar");

  if (bdImagens.empty() || vCompararImagens.empty())
  {
    std::cerr << "Erro: BD ou comparar estao vazios." << std::endl;
    return 1;
  }

  for (const auto &imgComparar : vCompararImagens)
  {
    processAndPrintSimilarities(imgComparar, bdImagens);
  }


  /* ARVORE */
  std::cout << "--------------------------------------------------------\n\n";
  std::cout << "  KDTree\n\n";
  std::cout << "--------------------------------------------------------\n\n";

  processWithKDTree(folderBD, folderComparar);

  /* TABELA COMPLETA DAS IMGS */
  if (!vCompararImagens.empty() && !bdImagens.empty())
  {
      printTable(vCompararImagens, bdImagens, vCompararImagens.size());
  }
  
  
  return 0;
}
