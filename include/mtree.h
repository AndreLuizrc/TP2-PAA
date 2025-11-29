#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <queue>

// Forward declarations
class MTreeNode;

/**
 * @brief Configuracoes da M-Tree
 */
struct MTreeConfig {
    int maxCapacity = 50;          // M: maximo de entradas por no
    int minCapacity = 25;          // m: minimo de entradas (M/2)

    enum class PromotionPolicy {
        RANDOM,                     // Selecao aleatoria O(1)
        SAMPLING                    // Amostragem aleatoria O(s^2)
    };
    PromotionPolicy promotionPolicy = PromotionPolicy::SAMPLING;
    int samplingSize = 10;          // Tamanho da amostra para SAMPLING

    enum class PartitionPolicy {
        BALANCED,                   // Distribuicao equilibrada
        MIN_RAD                     // Minimiza raio de cobertura
    };
    PartitionPolicy partitionPolicy = PartitionPolicy::MIN_RAD;
};

/**
 * @brief Entrada em no folha
 */
struct LeafEntry {
    cv::Mat feature;               // Vetor de caracteristicas (HOG)
    std::string objectId;          // Identificador unico (nome da imagem)
    double distanceToParent;       // d(O, O_r) - distancia ao roteador pai

    LeafEntry() : distanceToParent(0.0) {}
    LeafEntry(const cv::Mat& f, const std::string& id, double dist)
        : feature(f.clone()), objectId(id), distanceToParent(dist) {}
};

/**
 * @brief Entrada em no interno (routing entry)
 */
struct RoutingEntry {
    cv::Mat routingObject;         // Objeto roteador (pivo)
    double coveringRadius;         // Raio de cobertura r(O_r)
    double distanceToParent;       // d(O_r, O_p) - distancia ao roteador pai
    std::shared_ptr<MTreeNode> subtree;  // Ponteiro para subarvore

    RoutingEntry() : coveringRadius(0.0), distanceToParent(0.0) {}
    RoutingEntry(const cv::Mat& obj, double radius, double distParent, std::shared_ptr<MTreeNode> sub)
        : routingObject(obj.clone()), coveringRadius(radius),
          distanceToParent(distParent), subtree(sub) {}
};

/**
 * @brief No da M-Tree (pode ser interno ou folha)
 */
class MTreeNode {
public:
    MTreeNode(bool isLeafNode = true) : leaf(isLeafNode) {}

    bool isLeaf() const { return leaf; }
    void setLeaf(bool isLeafNode) { leaf = isLeafNode; }

    // Entradas do no
    std::vector<LeafEntry> leafEntries;       // Usado se isLeaf() == true
    std::vector<RoutingEntry> routingEntries; // Usado se isLeaf() == false

    // Objeto roteador do pai (para calculo de distancia)
    cv::Mat parentRoutingObject;

    size_t size() const {
        return leaf ? leafEntries.size() : routingEntries.size();
    }

private:
    bool leaf;
};

/**
 * @brief Resultado de busca
 */
struct MTreeSearchResult {
    std::string objectId;          // Nome/identificador do objeto
    double distance;               // Distancia ao query

    MTreeSearchResult() : distance(0.0) {}
    MTreeSearchResult(const std::string& id, double dist) : objectId(id), distance(dist) {}

    bool operator<(const MTreeSearchResult& other) const {
        return distance < other.distance;
    }
    bool operator>(const MTreeSearchResult& other) const {
        return distance > other.distance;
    }
};

/**
 * @brief M-Tree: Arvore Metrica para indexacao de espacos metricos
 *
 * Implementacao baseada em:
 * Ciaccia, P., Patella, M., & Zezula, P. (1997). M-tree: An Efficient
 * Access Method for Similarity Search in Metric Spaces. VLDB.
 */
class MTree {
public:
    // Tipo da funcao de distancia: d(a, b) -> double
    using DistanceFunction = std::function<double(const cv::Mat&, const cv::Mat&)>;

    /**
     * @brief Construtor
     * @param distFunc Funcao de distancia metrica
     * @param config Configuracoes da arvore
     */
    explicit MTree(DistanceFunction distFunc, MTreeConfig config = MTreeConfig{});

    /**
     * @brief Destrutor
     */
    ~MTree() = default;

    // ==================== Construcao ====================

    /**
     * @brief Constroi a arvore a partir de um diretorio de imagens
     * @param dirPath Caminho do diretorio com imagens
     */
    void buildFromDirectory(const std::string& dirPath);

    /**
     * @brief Insere um unico objeto na arvore
     * @param feature Vetor de caracteristicas
     * @param objectId Identificador unico
     */
    void insert(const cv::Mat& feature, const std::string& objectId);

    // ==================== Busca ====================

    /**
     * @brief Busca por raio: todos objetos com d(Q, O) <= radius
     * @param query Vetor de caracteristicas da consulta
     * @param radius Raio de busca
     * @return Vetor de resultados ordenados por distancia
     */
    std::vector<MTreeSearchResult> rangeSearch(const cv::Mat& query, double radius) const;

    /**
     * @brief Busca k vizinhos mais proximos
     * @param query Vetor de caracteristicas da consulta
     * @param k Numero de vizinhos
     * @return Vetor com os k vizinhos mais proximos
     */
    std::vector<MTreeSearchResult> knnSearch(const cv::Mat& query, int k) const;

    /**
     * @brief Busca a partir de uma imagem (carrega e vetoriza)
     * @param imagePath Caminho da imagem de consulta
     * @param k Numero de vizinhos (-1 para todos)
     * @return Vetor de pares (nome, similaridade cosseno)
     */
    std::vector<std::pair<std::string, double>> queryFromImage(
        const std::string& imagePath, int k = -1) const;

    // ==================== Utilitarios ====================

    /**
     * @brief Retorna o numero de objetos indexados
     */
    size_t size() const { return objectCount; }

    /**
     * @brief Verifica se a arvore esta vazia
     */
    bool empty() const { return root == nullptr; }

    /**
     * @brief Retorna a altura da arvore
     */
    int height() const;

    /**
     * @brief Limpa a arvore
     */
    void clear();

    /**
     * @brief Imprime estatisticas da arvore
     */
    void printStats() const;

    // ==================== Funcoes de Distancia Predefinidas ====================

    /**
     * @brief Distancia Euclidiana
     */
    static double euclideanDistance(const cv::Mat& a, const cv::Mat& b);

    /**
     * @brief Distancia derivada da similaridade cosseno
     * d(a,b) = sqrt(2(1 - cos(a,b)))
     */
    static double cosineDistance(const cv::Mat& a, const cv::Mat& b);
    
private:
    // ==================== Membros Privados ====================

    std::shared_ptr<MTreeNode> root;
    DistanceFunction distanceFunc;
    MTreeConfig config;
    size_t objectCount = 0;

    // ==================== Metodos Privados ====================

    // Insercao
    struct InsertResult {
        bool split;
        RoutingEntry newEntry;  // Nova entrada se houve split
    };

    InsertResult insertRecursive(std::shared_ptr<MTreeNode>& node,
                                  const cv::Mat& feature,
                                  const std::string& objectId,
                                  double distToParentRouting);

    int chooseSubtree(const std::shared_ptr<MTreeNode>& node, const cv::Mat& feature);

    // Split
    void splitLeafNode(std::shared_ptr<MTreeNode>& node,
                       std::shared_ptr<MTreeNode>& newNode,
                       cv::Mat& promotedPivot1,
                       cv::Mat& promotedPivot2);

    void splitInternalNode(std::shared_ptr<MTreeNode>& node,
                           std::shared_ptr<MTreeNode>& newNode,
                           cv::Mat& promotedPivot1,
                           cv::Mat& promotedPivot2);

    // Promocao
    std::pair<int, int> promoteRandom(int numEntries);
    std::pair<int, int> promoteSampling(const std::vector<cv::Mat>& objects);

    // Busca
    void rangeSearchRecursive(const std::shared_ptr<MTreeNode>& node,
                              const cv::Mat& query,
                              double radius,
                              double distToParentRouting,
                              std::vector<MTreeSearchResult>& results) const;

    // Utilitarios
    int computeHeight(const std::shared_ptr<MTreeNode>& node) const;
    void collectStats(const std::shared_ptr<MTreeNode>& node,
                      int& nodeCount, int& leafCount,
                      int& totalEntries, double& totalRadius) const;
};
