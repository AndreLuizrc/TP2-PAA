# M-Tree: Estrutura de Indexação para Espaços Métricos

## Sumário

1. [Fundamentos Teóricos](#1-fundamentos-teóricos)
2. [Estrutura de Dados](#2-estrutura-de-dados)
3. [Algoritmos Principais](#3-algoritmos-principais)
4. [Especificação C++](#4-especificação-c)
5. [Tarefas de Implementação](#5-tarefas-de-implementação)
6. [Critérios de Aceite](#6-critérios-de-aceite)
7. [Referências](#7-referências)

---

## 1. Fundamentos Teóricos

### 1.1 Introdução

A **M-Tree** (Metric Tree) é uma estrutura de dados de indexação projetada para realizar buscas eficientes por similaridade em **espaços métricos** genéricos. Foi proposta por Ciaccia, Patella e Zezula em 1997 [1] como uma solução para o problema de indexação em espaços onde apenas uma função de distância métrica está disponível, sem a necessidade de coordenadas explícitas.

### 1.2 Espaços Métricos

Um **espaço métrico** é definido como um par `(M, d)` onde:
- `M` é um domínio de objetos
- `d: M × M → ℝ⁺` é uma função de distância que satisfaz as propriedades:

| Propriedade | Definição | Descrição |
|-------------|-----------|-----------|
| **Não-negatividade** | `d(x, y) ≥ 0` | Distâncias são sempre positivas ou zero |
| **Identidade** | `d(x, y) = 0 ⟺ x = y` | Distância zero apenas para o mesmo objeto |
| **Simetria** | `d(x, y) = d(y, x)` | Ordem dos argumentos não importa |
| **Desigualdade Triangular** | `d(x, z) ≤ d(x, y) + d(y, z)` | Caminho direto é sempre o mais curto |

### 1.3 Métricas de Distância para Imagens

No contexto deste projeto, utilizamos **vetores HOG** (Histogram of Oriented Gradients) normalizados via L2. A métrica mais adequada é:

**Distância Euclidiana:**
```
d(a, b) = ||a - b||₂ = √(Σᵢ(aᵢ - bᵢ)²)
```

**Relação com Similaridade Cosseno:**
Para vetores L2-normalizados (||a|| = ||b|| = 1):
```
d²(a, b) = ||a - b||² = ||a||² + ||b||² - 2⟨a,b⟩ = 2(1 - cos(a,b))
```

Portanto:
```
d(a, b) = √(2(1 - cosineSimilarity(a, b)))
```

Esta relação permite converter entre distância euclidiana e similaridade cosseno, mantendo compatibilidade com o sistema existente.

### 1.4 Por que M-Tree?

| Aspecto | KD-Tree | M-Tree | Vantagem M-Tree |
|---------|---------|--------|-----------------|
| **Dimensionalidade** | Degradação exponencial em alta dimensão | Independente de dimensão explícita | ✓ Alta dimensão |
| **Métrica** | Requer coordenadas (Lp) | Qualquer métrica | ✓ Flexibilidade |
| **Espaço** | Euclidiano | Qualquer espaço métrico | ✓ Generalidade |
| **Balanceamento** | Pode desbalancear | Balanceada por construção | ✓ Previsibilidade |
| **Paginação** | Não otimizada | Otimizada para disco | ✓ Escalabilidade |

### 1.5 Estrutura Hierárquica

A M-Tree é uma árvore balanceada com as seguintes características:

```
                    [Raiz]
                   /      \
           [Nó Interno]   [Nó Interno]
            /    \           /    \
      [Folha] [Folha]   [Folha] [Folha]
       │ │ │   │ │ │     │ │     │ │ │
       ○ ○ ○   ○ ○ ○     ○ ○     ○ ○ ○
      objetos  objetos  objetos  objetos
```

**Propriedades:**
- Cada nó (exceto a raiz) possui um **objeto roteador** (routing object)
- Cada objeto roteador tem um **raio de cobertura** que delimita todos os objetos em sua subárvore
- Nós folha armazenam os objetos de dados reais
- A árvore é balanceada: todas as folhas estão no mesmo nível

---

## 2. Estrutura de Dados

### 2.1 Componentes Fundamentais

#### 2.1.1 Objeto Roteador (Routing Object)

O objeto roteador `O_r` armazena:
- `O_r`: objeto representativo (pivô)
- `r(O_r)`: **raio de cobertura** - distância máxima a qualquer objeto na subárvore
- `d(O_r, O_p)`: distância ao objeto roteador pai (para poda)
- `ptr(T(O_r))`: ponteiro para a subárvore raiz

```
     O_p (pai)
      │
      │ d(O_r, O_p)
      ↓
     O_r ←──── r(O_r) ────→
    /   \
   ○     ○  (objetos dentro do raio)
```

#### 2.1.2 Entrada de Nó Folha (Leaf Entry)

Cada entrada em nó folha armazena:
- `O_j`: objeto de dados
- `oid(O_j)`: identificador único do objeto
- `d(O_j, O_r)`: distância ao objeto roteador pai

#### 2.1.3 Estrutura do Nó

```
┌─────────────────────────────────────────────────────┐
│                    NÓ INTERNO                       │
├─────────────────────────────────────────────────────┤
│ Entry[0]: { O_r₀, r₀, d(O_r₀,O_p), ptr(T₀) }       │
│ Entry[1]: { O_r₁, r₁, d(O_r₁,O_p), ptr(T₁) }       │
│ ...                                                 │
│ Entry[m-1]: { O_rₘ₋₁, rₘ₋₁, d(O_rₘ₋₁,O_p), ptr(Tₘ₋₁)}│
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                    NÓ FOLHA                         │
├─────────────────────────────────────────────────────┤
│ Entry[0]: { O₀, oid₀, d(O₀, O_r) }                 │
│ Entry[1]: { O₁, oid₁, d(O₁, O_r) }                 │
│ ...                                                 │
│ Entry[n-1]: { Oₙ₋₁, oidₙ₋₁, d(Oₙ₋₁, O_r) }         │
└─────────────────────────────────────────────────────┘
```

### 2.2 Parâmetros da Árvore

| Parâmetro | Símbolo | Descrição | Valor Típico |
|-----------|---------|-----------|--------------|
| Capacidade máxima | `M` | Máximo de entradas por nó | 50-100 |
| Capacidade mínima | `m` | Mínimo de entradas (exceto raiz) | ⌈M/2⌉ |
| Dimensão dos vetores | `D` | Dimensão do espaço de features | ~2052 (HOG) |

### 2.3 Invariantes da M-Tree

1. **Cobertura**: Todo objeto `O` em `T(O_r)` satisfaz `d(O, O_r) ≤ r(O_r)`
2. **Balanceamento**: Todas as folhas estão no mesmo nível
3. **Ocupação**: Cada nó (exceto raiz) tem entre `m` e `M` entradas
4. **Consistência de distância**: `d(O_r, O_p)` está corretamente armazenado

---

## 3. Algoritmos Principais

### 3.1 Busca por Raio (Range Query)

**Objetivo**: Encontrar todos os objetos `O` tal que `d(Q, O) ≤ r_q`

```
Algorithm RangeSearch(Node N, Query Q, radius r_q, parent_dist d_parent)
    if N is leaf:
        for each entry (O, oid, d_to_parent) in N:
            // Poda por desigualdade triangular
            if |d_parent - d_to_parent| ≤ r_q:
                d_actual = d(Q, O)
                if d_actual ≤ r_q:
                    add (O, oid, d_actual) to results
    else:  // internal node
        for each entry (O_r, r, d_to_parent, ptr) in N:
            // Poda 1: desigualdade triangular
            if |d_parent - d_to_parent| ≤ r_q + r:
                d_to_routing = d(Q, O_r)
                // Poda 2: raio de cobertura
                if d_to_routing ≤ r_q + r:
                    RangeSearch(ptr, Q, r_q, d_to_routing)
    return results
```

**Complexidade**: O(log n) no melhor caso, O(n) no pior caso

### 3.2 Busca k-NN (k Nearest Neighbors)

**Objetivo**: Encontrar os `k` objetos mais próximos de `Q`

```
Algorithm kNNSearch(Node root, Query Q, int k)
    // Priority queue ordenada por limite inferior de distância
    PQ = new MinPriorityQueue()
    PQ.insert((root, 0, ∞))  // (node, d_min, d_parent)

    // Max-heap para os k melhores até agora
    NN = new MaxHeap(capacity=k)
    d_k = ∞  // distância do k-ésimo vizinho

    while not PQ.empty():
        (N, d_min, d_parent) = PQ.extractMin()

        // Poda: se d_min > d_k, não há candidatos melhores
        if d_min > d_k:
            break

        if N is leaf:
            for each entry (O, oid, d_to_parent) in N:
                // Poda por desigualdade triangular
                d_lower = |d_parent - d_to_parent|
                if d_lower ≤ d_k:
                    d_actual = d(Q, O)
                    if d_actual < d_k:
                        NN.insert((O, oid, d_actual))
                        if NN.size() > k:
                            NN.extractMax()
                        if NN.size() == k:
                            d_k = NN.peekMax().distance
        else:
            for each entry (O_r, r, d_to_parent, ptr) in N:
                d_lower = max(|d_parent - d_to_parent| - r, 0)
                if d_lower ≤ d_k:
                    d_to_routing = d(Q, O_r)
                    d_min_child = max(d_to_routing - r, 0)
                    if d_min_child ≤ d_k:
                        PQ.insert((ptr, d_min_child, d_to_routing))

    return NN.toSortedList()
```

**Complexidade**: O(log n) no melhor caso típico

### 3.3 Inserção

```
Algorithm Insert(Node N, Object O_new)
    if N is leaf:
        if N.size() < M:
            add O_new to N
            update parent covering radius if needed
        else:
            Split(N, O_new)
    else:
        // Escolher subárvore (múltiplas políticas possíveis)
        best_child = ChooseSubtree(N, O_new)
        Insert(best_child, O_new)
        update covering radius of best_child's routing object
```

### 3.4 Políticas de Escolha de Subárvore

| Política | Descrição | Trade-off |
|----------|-----------|-----------|
| **MIN_RADIUS_INCREASE** | Minimiza aumento do raio de cobertura | Bom para busca, construção lenta |
| **MIN_DISTANCE** | Escolhe roteador mais próximo | Construção rápida, árvore menos otimizada |
| **MIN_OVERLAP** | Minimiza sobreposição entre nós | Melhor para buscas, mais complexo |

### 3.5 Split de Nós

Quando um nó excede a capacidade `M`, deve ser dividido:

```
Algorithm Split(Node N, Object O_new)
    entries = N.entries ∪ {O_new}

    // 1. Selecionar dois objetos como novos roteadores (promoção)
    (O_r1, O_r2) = PromotionPolicy(entries)

    // 2. Particionar entradas entre os dois novos nós
    (N1, N2) = PartitionPolicy(entries, O_r1, O_r2)

    // 3. Atualizar raios de cobertura
    r1 = max(d(O, O_r1) for O in N1)
    r2 = max(d(O, O_r2) for O in N2)

    // 4. Propagar split para o pai (pode causar split recursivo)
    if N is root:
        create new root with entries (O_r1, r1, N1) and (O_r2, r2, N2)
    else:
        replace N with N1 in parent
        Insert (O_r2, r2, N2) into parent
```

#### 3.5.1 Políticas de Promoção

| Política | Complexidade | Qualidade |
|----------|--------------|-----------|
| **RANDOM** | O(1) | Baixa |
| **MAX_SPREAD** | O(n²) | Alta - maximiza separação |
| **MIN_MAX_RAD** | O(n²) | Alta - minimiza raio máximo |
| **SAMPLING** | O(s²), s << n | Média - amostragem aleatória |

#### 3.5.2 Políticas de Partição

| Política | Descrição |
|----------|-----------|
| **BALANCED** | Distribui igualmente entre os nós |
| **MIN_RAD** | Atribui ao roteador mais próximo |
| **MIN_OVERLAP** | Minimiza sobreposição das regiões |

### 3.6 Construção Bulk Loading

Para construção eficiente a partir de um conjunto de dados:

```
Algorithm BulkLoad(Objects[] data, int M)
    if data.size() ≤ M:
        return createLeafNode(data)

    // 1. Selecionar pivôs (amostragem ou clustering)
    pivots = SelectPivots(data, num_partitions)

    // 2. Particionar dados pelos pivôs
    partitions = PartitionByNearestPivot(data, pivots)

    // 3. Recursivamente construir subárvores
    children = []
    for each (pivot, partition) in zip(pivots, partitions):
        child = BulkLoad(partition, M)
        children.add((pivot, child))

    // 4. Construir nós internos bottom-up
    return buildInternalNodes(children, M)
```

---

## 4. Especificação C++

### 4.1 Arquivos

```
include/
    mtree.h              # Declaração da classe MTree
src/
    mtree.cpp            # Implementação
```

### 4.2 Estruturas de Dados

```cpp
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <queue>

// Forward declarations
class MTreeNode;
class MTree;

/**
 * @brief Configurações da M-Tree
 */
struct MTreeConfig {
    int maxCapacity = 50;          // M: máximo de entradas por nó
    int minCapacity = 25;          // m: mínimo de entradas (M/2)

    enum class PromotionPolicy {
        RANDOM,                     // Seleção aleatória O(1)
        MAX_SPREAD,                 // Maximiza distância entre pivôs O(n²)
        SAMPLING                    // Amostragem aleatória O(s²)
    };
    PromotionPolicy promotionPolicy = PromotionPolicy::SAMPLING;
    int samplingSize = 10;          // Tamanho da amostra para SAMPLING

    enum class PartitionPolicy {
        BALANCED,                   // Distribuição equilibrada
        MIN_RAD                     // Minimiza raio de cobertura
    };
    PartitionPolicy partitionPolicy = PartitionPolicy::MIN_RAD;
};

/**
 * @brief Entrada em nó folha
 */
struct LeafEntry {
    cv::Mat feature;               // Vetor de características (HOG)
    std::string objectId;          // Identificador único (nome da imagem)
    double distanceToParent;       // d(O, O_r) - distância ao roteador pai
};

/**
 * @brief Entrada em nó interno (routing entry)
 */
struct RoutingEntry {
    cv::Mat routingObject;         // Objeto roteador (pivô)
    double coveringRadius;         // Raio de cobertura r(O_r)
    double distanceToParent;       // d(O_r, O_p) - distância ao roteador pai
    std::shared_ptr<MTreeNode> subtree;  // Ponteiro para subárvore
};

/**
 * @brief Nó da M-Tree (pode ser interno ou folha)
 */
class MTreeNode {
public:
    bool isLeaf() const { return leaf; }

    // Entradas do nó
    std::vector<LeafEntry> leafEntries;       // Usado se isLeaf() == true
    std::vector<RoutingEntry> routingEntries; // Usado se isLeaf() == false

    // Referência ao roteador pai (nullptr para raiz)
    cv::Mat parentRoutingObject;

private:
    bool leaf = true;
    friend class MTree;
};

/**
 * @brief Resultado de busca
 */
struct MTreeSearchResult {
    std::string objectId;          // Nome/identificador do objeto
    double distance;               // Distância ao query

    bool operator<(const MTreeSearchResult& other) const {
        return distance < other.distance;
    }
    bool operator>(const MTreeSearchResult& other) const {
        return distance > other.distance;
    }
};

/**
 * @brief M-Tree: Árvore Métrica para indexação de espaços métricos
 *
 * Implementação baseada em:
 * Ciaccia, P., Patella, M., & Zezula, P. (1997). M-tree: An Efficient
 * Access Method for Similarity Search in Metric Spaces. VLDB.
 */
class MTree {
public:
    // Tipo da função de distância: d(a, b) -> double
    using DistanceFunction = std::function<double(const cv::Mat&, const cv::Mat&)>;

    /**
     * @brief Construtor
     * @param distFunc Função de distância métrica
     * @param config Configurações da árvore
     */
    explicit MTree(DistanceFunction distFunc, MTreeConfig config = MTreeConfig{});

    /**
     * @brief Destrutor
     */
    ~MTree() = default;

    // ==================== Construção ====================

    /**
     * @brief Constrói a árvore a partir de um diretório de imagens
     * @param dirPath Caminho do diretório com imagens
     */
    void buildFromDirectory(const std::string& dirPath);

    /**
     * @brief Insere um único objeto na árvore
     * @param feature Vetor de características
     * @param objectId Identificador único
     */
    void insert(const cv::Mat& feature, const std::string& objectId);

    /**
     * @brief Construção em lote (bulk loading) - mais eficiente
     * @param features Vetores de características
     * @param objectIds Identificadores correspondentes
     */
    void bulkLoad(const std::vector<cv::Mat>& features,
                  const std::vector<std::string>& objectIds);

    // ==================== Busca ====================

    /**
     * @brief Busca por raio: todos objetos com d(Q, O) ≤ radius
     * @param query Vetor de características da consulta
     * @param radius Raio de busca
     * @return Vetor de resultados ordenados por distância
     */
    std::vector<MTreeSearchResult> rangeSearch(const cv::Mat& query,
                                                double radius) const;

    /**
     * @brief Busca k vizinhos mais próximos
     * @param query Vetor de características da consulta
     * @param k Número de vizinhos
     * @return Vetor com os k vizinhos mais próximos
     */
    std::vector<MTreeSearchResult> knnSearch(const cv::Mat& query, int k) const;

    /**
     * @brief Busca a partir de uma imagem (carrega e vetoriza)
     * @param imagePath Caminho da imagem de consulta
     * @param k Número de vizinhos
     * @return Vetor com os k vizinhos mais próximos
     */
    std::vector<MTreeSearchResult> queryFromImage(const std::string& imagePath,
                                                   int k) const;

    // ==================== Utilitários ====================

    /**
     * @brief Retorna o número de objetos indexados
     */
    size_t size() const { return objectCount; }

    /**
     * @brief Verifica se a árvore está vazia
     */
    bool empty() const { return root == nullptr; }

    /**
     * @brief Retorna a altura da árvore
     */
    int height() const;

    /**
     * @brief Limpa a árvore
     */
    void clear();

    /**
     * @brief Imprime estatísticas da árvore
     */
    void printStats() const;

    // ==================== Funções de Distância Predefinidas ====================

    /**
     * @brief Distância Euclidiana
     */
    static double euclideanDistance(const cv::Mat& a, const cv::Mat& b);

    /**
     * @brief Distância derivada da similaridade cosseno
     * d(a,b) = sqrt(2(1 - cos(a,b)))
     */
    static double cosineDistance(const cv::Mat& a, const cv::Mat& b);

private:
    // ==================== Membros Privados ====================

    std::shared_ptr<MTreeNode> root;
    DistanceFunction distance;
    MTreeConfig config;
    size_t objectCount = 0;

    // ==================== Métodos Privados ====================

    // Inserção
    void insertRecursive(std::shared_ptr<MTreeNode> node,
                         const cv::Mat& feature,
                         const std::string& objectId,
                         double distToParent);

    std::shared_ptr<MTreeNode> chooseSubtree(std::shared_ptr<MTreeNode> node,
                                              const cv::Mat& feature);

    void split(std::shared_ptr<MTreeNode> node);

    // Promoção
    std::pair<int, int> promoteRandom(const std::vector<cv::Mat>& objects);
    std::pair<int, int> promoteMaxSpread(const std::vector<cv::Mat>& objects);
    std::pair<int, int> promoteSampling(const std::vector<cv::Mat>& objects);

    // Partição
    void partitionBalanced(std::vector<LeafEntry>& entries,
                           const cv::Mat& pivot1, const cv::Mat& pivot2,
                           std::vector<LeafEntry>& group1,
                           std::vector<LeafEntry>& group2);

    void partitionMinRad(std::vector<LeafEntry>& entries,
                         const cv::Mat& pivot1, const cv::Mat& pivot2,
                         std::vector<LeafEntry>& group1,
                         std::vector<LeafEntry>& group2);

    // Busca
    void rangeSearchRecursive(const std::shared_ptr<MTreeNode>& node,
                              const cv::Mat& query,
                              double radius,
                              double distToParent,
                              std::vector<MTreeSearchResult>& results) const;

    void knnSearchRecursive(const std::shared_ptr<MTreeNode>& node,
                            const cv::Mat& query,
                            int k,
                            double distToParent,
                            std::priority_queue<MTreeSearchResult>& results,
                            double& currentRadius) const;

    // Utilitários
    int computeHeight(const std::shared_ptr<MTreeNode>& node) const;
    void collectStats(const std::shared_ptr<MTreeNode>& node,
                      int& nodeCount, int& leafCount,
                      double& avgOccupancy) const;
};
```

### 4.3 Interface Pública Resumida

```cpp
// Construção
MTree tree(MTree::cosineDistance);      // ou euclideanDistance
tree.buildFromDirectory("imgs/BD");      // construção a partir de diretório

// Busca
auto results = tree.knnSearch(queryFeature, 5);     // 5 vizinhos mais próximos
auto results = tree.rangeSearch(queryFeature, 0.5); // raio 0.5
auto results = tree.queryFromImage("query.jpg", 5); // busca a partir de imagem

// Estatísticas
tree.printStats();
```

---

## 5. Tarefas de Implementação

### 5.1 Fase 1: Estruturas Base

#### Tarefa 1.1: Criar arquivos header e source
- [ ] Criar `include/mtree.h` com declarações
- [ ] Criar `src/mtree.cpp` com implementação base
- [ ] Atualizar `Makefile` para incluir novos arquivos

#### Tarefa 1.2: Implementar estruturas de dados
- [ ] Implementar `MTreeConfig`
- [ ] Implementar `LeafEntry` e `RoutingEntry`
- [ ] Implementar `MTreeNode`
- [ ] Implementar `MTreeSearchResult`
- [ ] Implementar construtor e destrutor de `MTree`

#### Tarefa 1.3: Implementar funções de distância
- [ ] Implementar `euclideanDistance()`
- [ ] Implementar `cosineDistance()`
- [ ] Criar testes unitários para funções de distância

### 5.2 Fase 2: Inserção e Construção

#### Tarefa 2.1: Implementar inserção básica
- [ ] Implementar `insert()` - inserção de um objeto
- [ ] Implementar `insertRecursive()` - recursão na árvore
- [ ] Implementar `chooseSubtree()` - seleção de subárvore

#### Tarefa 2.2: Implementar políticas de promoção
- [ ] Implementar `promoteRandom()`
- [ ] Implementar `promoteMaxSpread()`
- [ ] Implementar `promoteSampling()`

#### Tarefa 2.3: Implementar políticas de partição
- [ ] Implementar `partitionBalanced()`
- [ ] Implementar `partitionMinRad()`

#### Tarefa 2.4: Implementar split de nós
- [ ] Implementar `split()` para nós folha
- [ ] Implementar `split()` para nós internos
- [ ] Implementar propagação de split (split cascading)

#### Tarefa 2.5: Implementar bulk loading
- [ ] Implementar `bulkLoad()` para construção eficiente
- [ ] Implementar `buildFromDirectory()` integrando com `vectorization()`

### 5.3 Fase 3: Algoritmos de Busca

#### Tarefa 3.1: Implementar busca por raio
- [ ] Implementar `rangeSearch()`
- [ ] Implementar `rangeSearchRecursive()`
- [ ] Implementar poda por desigualdade triangular

#### Tarefa 3.2: Implementar busca k-NN
- [ ] Implementar `knnSearch()` com priority queue
- [ ] Implementar `knnSearchRecursive()`
- [ ] Implementar poda incremental por raio atual

#### Tarefa 3.3: Implementar busca por imagem
- [ ] Implementar `queryFromImage()` integrando com sistema existente
- [ ] Reutilizar `vectorization()` de `image_vectorization.h`

### 5.4 Fase 4: Utilitários e Integração

#### Tarefa 4.1: Implementar utilitários
- [ ] Implementar `size()`, `empty()`, `height()`
- [ ] Implementar `clear()`
- [ ] Implementar `printStats()`
- [ ] Implementar `computeHeight()` e `collectStats()`

#### Tarefa 4.2: Integração com main.cpp
- [ ] Adicionar inclusão de `mtree.h` em `main.cpp`
- [ ] Criar função `processMTree()` similar a `processListQuickSort()`
- [ ] Adicionar logging para benchmark em formato CSV

#### Tarefa 4.3: Atualizar sistema de build
- [ ] Atualizar `Makefile` com novas dependências
- [ ] Atualizar `CMakeLists.txt` se necessário
- [ ] Verificar compilação sem warnings

### 5.5 Fase 5: Testes e Validação

#### Tarefa 5.1: Testes unitários
- [ ] Testar inserção e recuperação de objetos
- [ ] Testar range query com casos conhecidos
- [ ] Testar k-NN com casos conhecidos
- [ ] Verificar invariantes da árvore após operações

#### Tarefa 5.2: Testes de integração
- [ ] Testar com conjunto de imagens existente (15 imagens)
- [ ] Comparar resultados com busca linear
- [ ] Verificar ordenação consistente dos resultados

#### Tarefa 5.3: Benchmarking
- [ ] Executar benchmark comparativo (Linear, KD-Tree, M-Tree)
- [ ] Gerar logs em `log/mtree_log.csv`
- [ ] Documentar resultados e análise de complexidade

---

## 6. Critérios de Aceite

### 6.1 Funcionalidade

| ID | Critério | Verificação |
|----|----------|-------------|
| **F1** | A M-Tree indexa corretamente todas as imagens do diretório | `tree.size()` == número de imagens |
| **F2** | k-NN retorna exatamente k resultados (ou menos se n < k) | `results.size()` == min(k, n) |
| **F3** | Range query retorna todos objetos dentro do raio | Verificar manualmente com distância |
| **F4** | Resultados ordenados por distância crescente | `results[i].distance <= results[i+1].distance` |
| **F5** | Resultados de k-NN correspondem à busca linear | Comparar top-k com sort linear |
| **F6** | Distâncias calculadas corretamente | Verificar `cosineDistance` com `cosineSimilarity` |

### 6.2 Corretude Estrutural

| ID | Critério | Verificação |
|----|----------|-------------|
| **C1** | Árvore balanceada | Todas as folhas no mesmo nível |
| **C2** | Raios de cobertura corretos | `d(O, O_r) <= r(O_r)` para todo O em T(O_r) |
| **C3** | Distâncias aos pais corretas | `distanceToParent` consistente |
| **C4** | Capacidade respeitada | `m <= entries.size() <= M` para não-raiz |
| **C5** | Sem vazamento de memória | Valgrind/sanitizers limpos |

### 6.3 Desempenho

| ID | Critério | Verificação |
|----|----------|-------------|
| **P1** | Construção em O(n log n) | Tempo proporcional a n log n |
| **P2** | k-NN típico melhor que linear | Medir número de cálculos de distância |
| **P3** | Poda efetiva | Menos de 50% dos nós visitados em k-NN |
| **P4** | Memória O(n) | Sem overhead excessivo por objeto |

### 6.4 Integração

| ID | Critério | Verificação |
|----|----------|-------------|
| **I1** | Compila sem warnings | `make` com `-Wall -Wextra` limpo |
| **I2** | Integra com `main.cpp` | Executa junto com outros algoritmos |
| **I3** | Gera logs compatíveis | CSV com mesmo formato dos existentes |
| **I4** | Reutiliza infraestrutura | Usa `vectorization()`, `cosineSimilarity()` |

### 6.5 Documentação

| ID | Critério | Verificação |
|----|----------|-------------|
| **D1** | Código comentado | Funções públicas documentadas |
| **D2** | Complexidades documentadas | Big-O para operações principais |
| **D3** | Exemplos de uso | Demonstração no `main.cpp` |

### 6.6 Checklist Final

```markdown
## Checklist de Aceite

### Compilação e Build
- [ ] `make clean && make` executa sem erros
- [ ] `make run` executa a aplicação completa
- [ ] Nenhum warning de compilação

### Funcionalidade
- [ ] M-Tree indexa 15 imagens do BD
- [ ] k-NN(5) retorna 5 resultados corretos
- [ ] Resultados correspondem à busca linear
- [ ] Range search funciona corretamente

### Estrutura
- [ ] Árvore está balanceada
- [ ] Invariantes de raio respeitados
- [ ] Sem vazamento de memória

### Benchmark
- [ ] Log gerado em `log/mtree_log.csv`
- [ ] Tempo de busca medido
- [ ] Comparação com outros métodos documentada

### Integração
- [ ] Funciona com imagens existentes
- [ ] Reutiliza código de vetorização
- [ ] Interface consistente com KD-Tree
```

---

## 7. Referências

### 7.1 Artigos Fundamentais

1. **Ciaccia, P., Patella, M., & Zezula, P.** (1997). *M-tree: An Efficient Access Method for Similarity Search in Metric Spaces*. Proceedings of the 23rd VLDB Conference, Athens, Greece.
   - Paper original da M-Tree
   - Define estrutura, algoritmos e análise de complexidade

2. **Ciaccia, P., Patella, M., & Zezula, P.** (1998). *A Cost Model for Similarity Queries in Metric Spaces*. Proceedings of the 17th ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems (PODS).
   - Modelo de custo para otimização de consultas

3. **Zezula, P., Amato, G., Dohnal, V., & Batko, M.** (2006). *Similarity Search: The Metric Space Approach*. Springer.
   - Livro completo sobre busca por similaridade
   - Capítulos dedicados a M-Tree e variantes

### 7.2 Recursos Complementares

4. **Dalal, N., & Triggs, B.** (2005). *Histograms of Oriented Gradients for Human Detection*. CVPR.
   - Descritor HOG utilizado para vetorização de imagens

5. **Samet, H.** (2006). *Foundations of Multidimensional and Metric Data Structures*. Morgan Kaufmann.
   - Referência para estruturas espaciais e métricas

### 7.3 Implementações de Referência

- **SISAP Library**: http://www.sisap.org/
  - Biblioteca de referência para indexação métrica

- **OpenCV**: https://opencv.org/
  - Utilizado para processamento de imagens e HOG

---

## Histórico de Versões

| Versão | Data | Autor | Descrição |
|--------|------|-------|-----------|
| 1.0 | 2024-11-29 | Claude (Senior CS Researcher) | Documento inicial |

---

*Este documento serve como especificação técnica completa para implementação da M-Tree no contexto do projeto TP2-PAA de análise comparativa de estruturas de indexação para busca por similaridade em imagens.*
