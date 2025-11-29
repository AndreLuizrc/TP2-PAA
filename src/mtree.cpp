#include "mtree.h"
#include "image_vectorization.h"

#include <filesystem>
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

namespace fs = std::filesystem;

// ============================================================================
// Funcoes de Distancia
// ============================================================================

double MTree::euclideanDistance(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat diff = a - b;
    return cv::norm(diff, cv::NORM_L2);
}

double MTree::cosineDistance(const cv::Mat& a, const cv::Mat& b) {
    // Para vetores L2-normalizados: d(a,b) = sqrt(2(1 - cos(a,b)))
    double cosine = a.dot(b) / (cv::norm(a) * cv::norm(b) + 1e-10);
    // Clamp para evitar problemas numericos
    cosine = std::max(-1.0, std::min(1.0, cosine));
    return std::sqrt(2.0 * (1.0 - cosine));
}

// ============================================================================
// Construtor
// ============================================================================

MTree::MTree(DistanceFunction distFunc, MTreeConfig cfg)
    : root(nullptr), distanceFunc(distFunc), config(cfg), objectCount(0) {
}

// ============================================================================
// Utilitarios internos
// ============================================================================

static bool isImageExtension(const std::string& ext) {
    if (ext.empty()) return false;
    std::string e = ext;
    std::transform(e.begin(), e.end(), e.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return (e == ".png" || e == ".jpg" || e == ".jpeg");
}

// ============================================================================
// Construcao a partir de diretorio
// ============================================================================

void MTree::buildFromDirectory(const std::string& dirPath) {
    clear();

    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (!entry.is_regular_file()) continue;
        if (!isImageExtension(entry.path().extension().string())) continue;

        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (img.empty()) continue;

        auto vecData = vectorization(img, entry.path().filename().string());
        insert(vecData.vector, vecData.name);
    }
}

// ============================================================================
// Insercao
// ============================================================================

void MTree::insert(const cv::Mat& feature, const std::string& objectId) {
    if (root == nullptr) {
        // Arvore vazia: criar no folha raiz
        root = std::make_shared<MTreeNode>(true);
        root->leafEntries.emplace_back(feature, objectId, 0.0);
        objectCount++;
        return;
    }

    // Inserir recursivamente
    InsertResult result = insertRecursive(root, feature, objectId, 0.0);

    if (result.split) {
        // Raiz sofreu split: criar nova raiz
        auto newRoot = std::make_shared<MTreeNode>(false);

        // Primeiro filho: raiz antiga (com pivo promovido 1)
        RoutingEntry entry1;
        entry1.routingObject = root->isLeaf() ?
            root->leafEntries[0].feature.clone() :
            root->routingEntries[0].routingObject.clone();
        entry1.subtree = root;
        entry1.distanceToParent = 0.0;

        // Calcular raio de cobertura para entry1
        double maxDist1 = 0.0;
        if (root->isLeaf()) {
            for (const auto& le : root->leafEntries) {
                double d = distanceFunc(entry1.routingObject, le.feature);
                maxDist1 = std::max(maxDist1, d);
            }
        } else {
            for (const auto& re : root->routingEntries) {
                double d = distanceFunc(entry1.routingObject, re.routingObject) + re.coveringRadius;
                maxDist1 = std::max(maxDist1, d);
            }
        }
        entry1.coveringRadius = maxDist1;

        // Segundo filho: novo no do split
        RoutingEntry entry2 = result.newEntry;

        newRoot->routingEntries.push_back(entry1);
        newRoot->routingEntries.push_back(entry2);

        root = newRoot;
    }

    objectCount++;
}

MTree::InsertResult MTree::insertRecursive(std::shared_ptr<MTreeNode>& node,
                                            const cv::Mat& feature,
                                            const std::string& objectId,
                                            double distToParentRouting) {
    InsertResult result;
    result.split = false;

    if (node->isLeaf()) {
        // Inserir no no folha
        node->leafEntries.emplace_back(feature, objectId, distToParentRouting);

        // Verificar overflow
        if (static_cast<int>(node->leafEntries.size()) > config.maxCapacity) {
            // Split necessario
            auto newNode = std::make_shared<MTreeNode>(true);
            cv::Mat pivot1, pivot2;

            splitLeafNode(node, newNode, pivot1, pivot2);

            result.split = true;
            result.newEntry.routingObject = pivot2;
            result.newEntry.subtree = newNode;
            result.newEntry.distanceToParent = 0.0; // Sera calculado pelo pai

            // Calcular raio de cobertura
            double maxDist = 0.0;
            for (const auto& le : newNode->leafEntries) {
                double d = distanceFunc(pivot2, le.feature);
                maxDist = std::max(maxDist, d);
            }
            result.newEntry.coveringRadius = maxDist;
        }
    } else {
        // No interno: escolher subarvore
        int bestIdx = chooseSubtree(node, feature);
        auto& bestEntry = node->routingEntries[bestIdx];

        double distToRouting = distanceFunc(bestEntry.routingObject, feature);

        // Inserir recursivamente
        InsertResult childResult = insertRecursive(bestEntry.subtree, feature, objectId, distToRouting);

        // Atualizar raio de cobertura se necessario
        if (distToRouting > bestEntry.coveringRadius) {
            bestEntry.coveringRadius = distToRouting;
        }

        if (childResult.split) {
            // Filho sofreu split: adicionar nova entrada
            childResult.newEntry.distanceToParent =
                distanceFunc(node->parentRoutingObject.empty() ?
                             node->routingEntries[0].routingObject :
                             node->parentRoutingObject,
                             childResult.newEntry.routingObject);

            node->routingEntries.push_back(childResult.newEntry);

            // Verificar overflow no no interno
            if (static_cast<int>(node->routingEntries.size()) > config.maxCapacity) {
                auto newNode = std::make_shared<MTreeNode>(false);
                cv::Mat pivot1, pivot2;

                splitInternalNode(node, newNode, pivot1, pivot2);

                result.split = true;
                result.newEntry.routingObject = pivot2;
                result.newEntry.subtree = newNode;
                result.newEntry.distanceToParent = 0.0;

                // Calcular raio de cobertura
                double maxDist = 0.0;
                for (const auto& re : newNode->routingEntries) {
                    double d = distanceFunc(pivot2, re.routingObject) + re.coveringRadius;
                    maxDist = std::max(maxDist, d);
                }
                result.newEntry.coveringRadius = maxDist;
            }
        }
    }

    return result;
}

int MTree::chooseSubtree(const std::shared_ptr<MTreeNode>& node, const cv::Mat& feature) {
    // Estrategia: escolher entrada cujo raio precisa aumentar menos
    // Se couber em algum raio, escolher o mais proximo

    int bestIdx = 0;
    double bestIncrease = std::numeric_limits<double>::max();
    double bestDist = std::numeric_limits<double>::max();
    bool foundContaining = false;

    for (size_t i = 0; i < node->routingEntries.size(); i++) {
        const auto& entry = node->routingEntries[i];
        double dist = distanceFunc(entry.routingObject, feature);

        if (dist <= entry.coveringRadius) {
            // Feature esta dentro do raio
            if (!foundContaining || dist < bestDist) {
                foundContaining = true;
                bestIdx = static_cast<int>(i);
                bestDist = dist;
            }
        } else if (!foundContaining) {
            // Precisa aumentar o raio
            double increase = dist - entry.coveringRadius;
            if (increase < bestIncrease) {
                bestIncrease = increase;
                bestIdx = static_cast<int>(i);
            }
        }
    }

    return bestIdx;
}

// ============================================================================
// Split de Nos
// ============================================================================

std::pair<int, int> MTree::promoteRandom(int numEntries) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_int_distribution<> dist(0, numEntries - 1);
    int idx1 = dist(gen);
    int idx2;
    do {
        idx2 = dist(gen);
    } while (idx2 == idx1);

    return {idx1, idx2};
}

std::pair<int, int> MTree::promoteSampling(const std::vector<cv::Mat>& objects) {
    // Amostragem: pegar ate samplingSize elementos e escolher par mais distante
    int n = static_cast<int>(objects.size());
    int sampleSize = std::min(config.samplingSize, n);

    std::vector<int> indices(n);
    for (int i = 0; i < n; i++) indices[i] = i;

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    double maxDist = -1.0;
    int best1 = 0, best2 = 1;

    for (int i = 0; i < sampleSize; i++) {
        for (int j = i + 1; j < sampleSize; j++) {
            double d = distanceFunc(objects[indices[i]], objects[indices[j]]);
            if (d > maxDist) {
                maxDist = d;
                best1 = indices[i];
                best2 = indices[j];
            }
        }
    }

    return {best1, best2};
}

void MTree::splitLeafNode(std::shared_ptr<MTreeNode>& node,
                          std::shared_ptr<MTreeNode>& newNode,
                          cv::Mat& promotedPivot1,
                          cv::Mat& promotedPivot2) {
    // Coletar todos os objetos
    std::vector<cv::Mat> objects;
    for (const auto& le : node->leafEntries) {
        objects.push_back(le.feature);
    }

    // Selecionar pivos
    std::pair<int, int> pivots;
    if (config.promotionPolicy == MTreeConfig::PromotionPolicy::RANDOM) {
        pivots = promoteRandom(static_cast<int>(objects.size()));
    } else {
        pivots = promoteSampling(objects);
    }

    promotedPivot1 = objects[pivots.first].clone();
    promotedPivot2 = objects[pivots.second].clone();

    // Particionar: cada entrada vai para o pivo mais proximo
    std::vector<LeafEntry> entries1, entries2;

    for (const auto& le : node->leafEntries) {
        double d1 = distanceFunc(promotedPivot1, le.feature);
        double d2 = distanceFunc(promotedPivot2, le.feature);

        if (d1 <= d2) {
            entries1.emplace_back(le.feature, le.objectId, d1);
        } else {
            entries2.emplace_back(le.feature, le.objectId, d2);
        }
    }

    // Garantir que ambos os nos tenham pelo menos uma entrada
    if (entries1.empty()) {
        entries1.push_back(entries2.back());
        entries2.pop_back();
    } else if (entries2.empty()) {
        entries2.push_back(entries1.back());
        entries1.pop_back();
    }

    node->leafEntries = std::move(entries1);
    newNode->leafEntries = std::move(entries2);
}

void MTree::splitInternalNode(std::shared_ptr<MTreeNode>& node,
                              std::shared_ptr<MTreeNode>& newNode,
                              cv::Mat& promotedPivot1,
                              cv::Mat& promotedPivot2) {
    // Coletar todos os objetos roteadores
    std::vector<cv::Mat> objects;
    for (const auto& re : node->routingEntries) {
        objects.push_back(re.routingObject);
    }

    // Selecionar pivos
    std::pair<int, int> pivots;
    if (config.promotionPolicy == MTreeConfig::PromotionPolicy::RANDOM) {
        pivots = promoteRandom(static_cast<int>(objects.size()));
    } else {
        pivots = promoteSampling(objects);
    }

    promotedPivot1 = objects[pivots.first].clone();
    promotedPivot2 = objects[pivots.second].clone();

    // Particionar
    std::vector<RoutingEntry> entries1, entries2;

    for (auto& re : node->routingEntries) {
        double d1 = distanceFunc(promotedPivot1, re.routingObject);
        double d2 = distanceFunc(promotedPivot2, re.routingObject);

        if (d1 <= d2) {
            re.distanceToParent = d1;
            entries1.push_back(std::move(re));
        } else {
            re.distanceToParent = d2;
            entries2.push_back(std::move(re));
        }
    }

    // Garantir que ambos os nos tenham pelo menos uma entrada
    if (entries1.empty()) {
        entries1.push_back(std::move(entries2.back()));
        entries2.pop_back();
    } else if (entries2.empty()) {
        entries2.push_back(std::move(entries1.back()));
        entries1.pop_back();
    }

    node->routingEntries = std::move(entries1);
    newNode->routingEntries = std::move(entries2);
}

// ============================================================================
// Busca por Raio (Range Search)
// ============================================================================

std::vector<MTreeSearchResult> MTree::rangeSearch(const cv::Mat& query, double radius) const {
    std::vector<MTreeSearchResult> results;

    if (root == nullptr) return results;

    rangeSearchRecursive(root, query, radius, 0.0, results);

    // Ordenar por distancia
    std::sort(results.begin(), results.end());

    return results;
}

void MTree::rangeSearchRecursive(const std::shared_ptr<MTreeNode>& node,
                                  const cv::Mat& query,
                                  double radius,
                                  double distToParentRouting,
                                  std::vector<MTreeSearchResult>& results) const {
    if (node->isLeaf()) {
        for (const auto& le : node->leafEntries) {
            // Poda por desigualdade triangular
            double lowerBound = std::abs(distToParentRouting - le.distanceToParent);
            if (lowerBound > radius) continue;

            double dist = distanceFunc(query, le.feature);
            if (dist <= radius) {
                results.emplace_back(le.objectId, dist);
            }
        }
    } else {
        for (const auto& re : node->routingEntries) {
            // Poda 1: desigualdade triangular
            double lowerBound = std::abs(distToParentRouting - re.distanceToParent);
            if (lowerBound > radius + re.coveringRadius) continue;

            double distToRouting = distanceFunc(query, re.routingObject);

            // Poda 2: raio de cobertura
            if (distToRouting > radius + re.coveringRadius) continue;

            // Recursao
            rangeSearchRecursive(re.subtree, query, radius, distToRouting, results);
        }
    }
}

// ============================================================================
// Busca k-NN
// ============================================================================

std::vector<MTreeSearchResult> MTree::knnSearch(const cv::Mat& query, int k) const {
    std::vector<MTreeSearchResult> results;

    if (root == nullptr || k <= 0) return results;

    // Priority queue para nos a visitar (min-heap por distancia minima)
    using NodeEntry = std::tuple<double, double, std::shared_ptr<MTreeNode>>;
    // (distancia minima, distancia ao roteador pai, no)

    auto cmp = [](const NodeEntry& a, const NodeEntry& b) {
        return std::get<0>(a) > std::get<0>(b);  // min-heap
    };
    std::priority_queue<NodeEntry, std::vector<NodeEntry>, decltype(cmp)> pq(cmp);

    // Max-heap para os k melhores resultados
    std::priority_queue<MTreeSearchResult> kBest;

    double dk = std::numeric_limits<double>::max();  // Distancia do k-esimo vizinho

    pq.push({0.0, 0.0, root});

    while (!pq.empty()) {
        auto [dMin, dParent, node] = pq.top();
        pq.pop();

        // Poda: se distancia minima > dk, nao ha candidatos melhores
        if (dMin > dk) continue;

        if (node->isLeaf()) {
            for (const auto& le : node->leafEntries) {
                // Poda por desigualdade triangular
                double lowerBound = std::abs(dParent - le.distanceToParent);
                if (lowerBound > dk) continue;

                double dist = distanceFunc(query, le.feature);

                if (dist < dk || static_cast<int>(kBest.size()) < k) {
                    kBest.push(MTreeSearchResult(le.objectId, dist));
                    if (static_cast<int>(kBest.size()) > k) {
                        kBest.pop();
                    }
                    if (static_cast<int>(kBest.size()) == k) {
                        dk = kBest.top().distance;
                    }
                }
            }
        } else {
            for (const auto& re : node->routingEntries) {
                double distToRouting = distanceFunc(query, re.routingObject);
                double dMinChild = std::max(distToRouting - re.coveringRadius, 0.0);

                if (dMinChild <= dk) {
                    pq.push({dMinChild, distToRouting, re.subtree});
                }
            }
        }
    }

    // Extrair resultados do heap
    while (!kBest.empty()) {
        results.push_back(kBest.top());
        kBest.pop();
    }

    // Reverter para ordem crescente de distancia
    std::reverse(results.begin(), results.end());

    return results;
}

// ============================================================================
// Query a partir de imagem
// ============================================================================

std::vector<std::pair<std::string, double>> MTree::queryFromImage(
    const std::string& imagePath, int k) const {

    std::vector<std::pair<std::string, double>> results;

    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) return results;

    auto vecData = vectorization(img, imagePath);

    int numResults = (k <= 0) ? static_cast<int>(objectCount) : k;
    auto knnResults = knnSearch(vecData.vector, numResults);

    // Converter distancia para similaridade cosseno
    // d = sqrt(2(1 - cos)) => cos = 1 - d^2/2
    for (const auto& r : knnResults) {
        double cosine = 1.0 - (r.distance * r.distance) / 2.0;
        results.emplace_back(r.objectId, cosine);
    }

    return results;
}

// ============================================================================
// Utilitarios
// ============================================================================

void MTree::clear() {
    root = nullptr;
    objectCount = 0;
}

int MTree::height() const {
    return computeHeight(root);
}

int MTree::computeHeight(const std::shared_ptr<MTreeNode>& node) const {
    if (node == nullptr) return 0;
    if (node->isLeaf()) return 1;

    int maxChildHeight = 0;
    for (const auto& re : node->routingEntries) {
        maxChildHeight = std::max(maxChildHeight, computeHeight(re.subtree));
    }
    return 1 + maxChildHeight;
}

void MTree::collectStats(const std::shared_ptr<MTreeNode>& node,
                         int& nodeCount, int& leafCount,
                         int& totalEntries, double& totalRadius) const {
    if (node == nullptr) return;

    nodeCount++;

    if (node->isLeaf()) {
        leafCount++;
        totalEntries += static_cast<int>(node->leafEntries.size());
    } else {
        totalEntries += static_cast<int>(node->routingEntries.size());
        for (const auto& re : node->routingEntries) {
            totalRadius += re.coveringRadius;
            collectStats(re.subtree, nodeCount, leafCount, totalEntries, totalRadius);
        }
    }
}

void MTree::printStats() const {
    std::cout << "\n========== M-Tree Statistics ==========\n";
    std::cout << "Objects indexed: " << objectCount << "\n";
    std::cout << "Tree height: " << height() << "\n";
    std::cout << "Max capacity (M): " << config.maxCapacity << "\n";
    std::cout << "Min capacity (m): " << config.minCapacity << "\n";

    if (root != nullptr) {
        int nodeCount = 0, leafCount = 0, totalEntries = 0;
        double totalRadius = 0.0;
        collectStats(root, nodeCount, leafCount, totalEntries, totalRadius);

        std::cout << "Total nodes: " << nodeCount << "\n";
        std::cout << "Leaf nodes: " << leafCount << "\n";
        std::cout << "Internal nodes: " << (nodeCount - leafCount) << "\n";

        if (nodeCount > 0) {
            std::cout << "Avg entries/node: " << std::fixed << std::setprecision(2)
                      << static_cast<double>(totalEntries) / nodeCount << "\n";
        }
        if (nodeCount - leafCount > 0) {
            std::cout << "Avg covering radius: " << std::fixed << std::setprecision(4)
                      << totalRadius / (nodeCount - leafCount) << "\n";
        }
    }

    std::cout << "========================================\n\n";
}
