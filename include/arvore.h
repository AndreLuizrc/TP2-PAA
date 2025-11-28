#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <unordered_map>

// ponto 2D (H̄,S̄) + nome do arquivo
struct Ponto2D { float x, y; std::string name; };

class KDTree2D {
public:
  // monta SÓ a KD-tree (a partir de (H̄,S̄))
  void buildFromDir(const std::string& dirPath);

  // monta KD-tree + index de HOGs (nome -> vetor HOG)
  void buildIndex(const std::string& dirPath);

  // consulta: recebe caminho da imagem, retorna TODOS ordenados por cosseno/HOG
  // k = quantos candidatos pegar na KD antes do re-rank (k<=0 => todos)
  std::vector<std::pair<std::string,double>>
  queryAllFromImage(const std::string& imgPath, int k = -1) const;

  // helper público para extrair (H̄,S̄) normalizado [0,1]
  static Ponto2D hsMean(const cv::Mat& bgr, const std::string& name);

  ~KDTree2D(){ free(root); }

  // (opcional) acesso ao mapa de HOGs (nome -> vetor)
  const std::unordered_map<std::string, cv::Mat>& hogIndex() const { return hogBase; }

private:
  struct Node { Ponto2D p; Node *left=nullptr,*right=nullptr; int axis=0; };
  Node* root = nullptr;

  // base de HOGs para re-rank
  std::unordered_map<std::string, cv::Mat> hogBase;

  // --- KD internals ---
  static float dist2(const Ponto2D& a, float x, float y);
  Node* build(std::vector<Ponto2D>& pts, int l, int r, int depth);
  void knn(Node* n, float qx, float qy, int k,
           std::vector<std::pair<float,const Ponto2D*>>& heap) const;
  void free(Node* n);
};
