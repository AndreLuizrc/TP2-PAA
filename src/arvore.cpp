#include "arvore.h"
#include "image_vectorization.h"

#include <filesystem>
#include <algorithm>
#include <queue>
#include <cctype>

namespace fs = std::filesystem;

// -------- util --------
static inline bool isImgExt(const std::string& ext) {
  if (ext.empty()) return false;
  std::string e = ext;
  for (char& c : e) c = std::tolower(c);
  return (e==".png" || e==".jpg" || e==".jpeg");
}

// -------- KD helpers --------
float KDTree2D::dist2(const Ponto2D& a, float x, float y){
  float dx = a.x - x, dy = a.y - y; return dx*dx + dy*dy;
}

Ponto2D KDTree2D::hsMean(const cv::Mat& bgr, const std::string& name){
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  cv::Scalar m = cv::mean(hsv);               // m[0]=H (0..180), m[1]=S (0..255)
  return { float(m[0])/180.f, float(m[1])/255.f, name };
}

KDTree2D::Node* KDTree2D::build(std::vector<Ponto2D>& pts, int l, int r, int depth){
  if (l >= r) return nullptr;
  int axis = depth % 2;
  int mid  = (l + r) / 2;
  std::nth_element(pts.begin()+l, pts.begin()+mid, pts.begin()+r,
    [axis](const Ponto2D& a, const Ponto2D& b){
      return axis ? (a.y < b.y) : (a.x < b.x);
    }
  );
  Node* n = new Node();
  n->p = pts[mid];
  n->axis = axis;
  n->left  = build(pts, l, mid, depth+1);
  n->right = build(pts, mid+1, r, depth+1);
  return n;
}

void KDTree2D::knn(Node* n, float qx, float qy, int k,
                   std::vector<std::pair<float,const Ponto2D*>>& heap) const
{
  if (!n) return;
  float d2 = dist2(n->p, qx, qy);

  // max-heap (maior distância no topo) via vetor + heap ops
  if ((int)heap.size() < k) {
    heap.push_back({d2, &n->p});
    std::push_heap(heap.begin(), heap.end());
  } else if (d2 < heap.front().first) {
    std::pop_heap(heap.begin(), heap.end());
    heap.back() = {d2, &n->p};
    std::push_heap(heap.begin(), heap.end());
  }

  float split = n->axis ? (qy - n->p.y) : (qx - n->p.x);
  Node* first  = (split <= 0.f) ? n->left  : n->right;
  Node* second = (split <= 0.f) ? n->right : n->left;

  knn(first, qx, qy, k, heap);

  if ((int)heap.size() < k || split*split < heap.front().first)
    knn(second, qx, qy, k, heap);
}

void KDTree2D::free(Node* n){
  if (!n) return;
  free(n->left); free(n->right); delete n;
}

// -------- build-only KD --------
void KDTree2D::buildFromDir(const std::string& dirPath){
  std::vector<Ponto2D> pts;
  for (auto& p : fs::directory_iterator(dirPath)){
    if (!p.is_regular_file()) continue;
    if (!isImgExt(p.path().extension().string())) continue;
    cv::Mat img = cv::imread(p.path().string(), cv::IMREAD_COLOR);
    if (img.empty()) continue;
    pts.push_back( hsMean(img, p.path().filename().string()) );
  }
  // limpa árvore antiga (se houver)
  free(root); root = nullptr;
  if (!pts.empty()) root = build(pts, 0, (int)pts.size(), 0);
}

// -------- build KD + HOG index --------
void KDTree2D::buildIndex(const std::string& dirPath){
  buildFromDir(dirPath);               // KD (H̄,S̄)
  hogBase.clear(); hogBase.reserve(1024);
  for (auto& p : fs::directory_iterator(dirPath)){
    if (!p.is_regular_file()) continue;
    if (!isImgExt(p.path().extension().string())) continue;
    cv::Mat img = cv::imread(p.path().string(), cv::IMREAD_COLOR);
    if (img.empty()) continue;
    auto v = vectorization(img, p.path().filename().string()); // HOG 1xN (L2 norm)
    hogBase.emplace(v.name, v.vector);
  }
}

// -------- query (KD -> candidatos -> re-rank por HOG/cosseno) --------
std::vector<std::pair<std::string,double>>
KDTree2D::queryAllFromImage(const std::string& imgPath, int k) const {
  if (!root || hogBase.empty()) return {};

  cv::Mat qimg = cv::imread(imgPath, cv::IMREAD_COLOR);
  if (qimg.empty()) return {};

  // HOG da consulta
  auto qhog = vectorization(qimg, imgPath); // usa nome da consulta só como label

  // ponto 2D da consulta
  auto qp = hsMean(qimg, imgPath);

  // pega candidatos na KD (k<=0 => todos)
  int n = (int)hogBase.size();
  int kk = (k <= 0 || k > n) ? n : k;

  std::vector<std::pair<float,const Ponto2D*>> heap;
  heap.reserve(kk);
  std::make_heap(heap.begin(), heap.end()); // inicia max-heap vazio
  knn(root, qp.x, qp.y, kk, heap);

  // transforma heap em vetor ordenado por menor distância
  std::sort_heap(heap.begin(), heap.end()); // crescente por dist²

  // re-rank por cosineSimilarity (HOG completo)
  std::vector<std::pair<std::string,double>> out;
  out.reserve(heap.size());
  for (auto& e : heap){
    auto it = hogBase.find(e.second->name);
    if (it == hogBase.end()) continue;
    double cos = cosineSimilarity(qhog.vector, it->second);
    out.emplace_back(e.second->name, cos);
  }

  std::sort(out.begin(), out.end(),
            [](auto& a, auto& b){ return a.second > b.second; });
  return out;
}
