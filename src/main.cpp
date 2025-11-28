#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>
#include "arvore.h"
#include <opencv2/opencv.hpp>
#include "image_vectorization.h"
#include "my_header.h"

namespace fs = std::filesystem;

int main()
{
  const std::string folderBD = "imgs/BD";
  const std::string folderComparar = "imgs/comparar";

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
