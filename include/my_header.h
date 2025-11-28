#ifndef MY_HEADER_H
#define MY_HEADER_H

#include <vector>
#include <string>
#include "image_vectorization.h"

/** Estrutura para armazenar a similaridade de uma imagem do BD com uma de comparação
 * @param bdImage {ImageVectorData} Lista da Base de Dados - Representada num vetor
 * @param similarity {double} Similaridade entre as imagens comparadas
*/
struct SimilarityResult {
    ImageVectorData bdImage;
    double similarity;
};

std::vector<ImageVectorData> loadImages(const std::string& folderPath, const std::string& tipo);

/* PRINTAR TABELA GERAL */
void printHeader(const std::vector<ImageVectorData>& compararImagens);

void printRows(const std::vector<ImageVectorData>& bdImagens,
               const std::vector<ImageVectorData>& compararImagens);

void printFooter(size_t compararSize);

void printTable(const std::vector<ImageVectorData>& compararImagens,
                const std::vector<ImageVectorData>& bdImagens,
                size_t compararSize);

/* PROCESSAR E PRINTAR LISTA ORDENADA */
void processAndPrintSimilarities(const ImageVectorData& imgComparar, const std::vector<ImageVectorData>& bdImagens);

/* PROCESSAR E PRINTAR KD Tree */
void processWithKDTree(const std::string& folderBD, const std::string& folderComparar);

#endif // MY_HEADER_H