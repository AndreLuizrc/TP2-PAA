#pragma once
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include "image_vectorization.h"

class LSHIndex {
public:
    struct Result {
        std::string image_name;
        double similarity;
    };

    LSHIndex(
        int dim,
        int num_tables = 5,
        int num_planes = 16,
        unsigned int seed = 42
    ) :
        dim_(dim),
        num_tables_(num_tables),
        num_planes_(num_planes),
        rng_(seed),
        dist_(0.0, 1.0)
    {
        generateHyperplanes();
        tables_.resize(num_tables_);
    }

    // ----------------------------------------------------
    // Inserção no índice
    // ----------------------------------------------------
    void add(const ImageVectorData data) {
        cv::Mat normalized = data.vector;
        std::vector<std::string> hashes = hashVector(normalized);

        for (int t = 0; t < num_tables_; ++t) {
            tables_[t][hashes[t]].push_back({
                data.name,
                normalized.clone()
            });
        }
    }

    // ----------------------------------------------------
    // Consulta
    // ----------------------------------------------------
    std::vector<Result> query(const cv::Mat& q, int top_k = 5, int max_candidates = 0) {
        // cv::Mat normalized = l2_normalize(q);
        std::vector<std::string> hashes = hashVector(q);

        // juntar candidatos sem repetir
        std::unordered_map<std::string, cv::Mat> candidates;

        for (int t = 0; t < num_tables_; ++t) {
            auto it = tables_[t].find(hashes[t]);
            if (it == tables_[t].end())
                continue;

            for (auto& item : it->second) {
                if (!candidates.count(item.name))
                    candidates[item.name] = item.vector;
            }
        }

        // limitar número de candidatos
        std::vector<std::pair<std::string, cv::Mat>> items;
        for (auto& kv : candidates)
            items.push_back(kv);

        if (max_candidates > 0 && items.size() > (size_t)max_candidates)
            items.resize(max_candidates);

        // calcular similaridade real
        std::vector<Result> results;
        for (auto& item : items) {
            double sim = cosineSimilarity(q, item.second);
            results.push_back({ item.first, sim });
        }

        std::sort(results.begin(), results.end(),
            [](auto& a, auto& b) { return a.similarity > b.similarity; }
        );

        if (results.size() > (size_t)top_k)
            results.resize(top_k);

        return results;
    }

private:
    int dim_;
    int num_tables_;
    int num_planes_;

    std::mt19937 rng_;
    std::normal_distribution<double> dist_;

    // hyperplanes[t][p] = cv::Mat 1xDIM
    std::vector<std::vector<cv::Mat>> hyperplanes_;

    // Cada tabela: hash -> lista de (nome + vetor)
    struct StoredData {
        std::string name;
        cv::Mat vector;
    };
    std::vector<std::unordered_map<std::string, std::vector<StoredData>>> tables_;

    // ----------------------------------------------------
    // Geração dos hiperplanos aleatórios
    // ----------------------------------------------------
    void generateHyperplanes() {
        hyperplanes_.resize(num_tables_);

        for (int t = 0; t < num_tables_; ++t) {
            hyperplanes_[t].resize(num_planes_);

            for (int p = 0; p < num_planes_; ++p) {
                cv::Mat w(1, dim_, CV_64F);
                for (int d = 0; d < dim_; ++d)
                    w.at<double>(0, d) = dist_(rng_);
                hyperplanes_[t][p] = w;
            }
        }
    }

    // ----------------------------------------------------
    // Hashing do vetor pela projeção em hiperplanos
    // ----------------------------------------------------
    std::vector<std::string> hashVector(const cv::Mat& v) {
        std::vector<std::string> hashes;
        hashes.reserve(num_tables_);

        for (int t = 0; t < num_tables_; ++t) {
            std::string bits;
            bits.reserve(num_planes_);

            for (int p = 0; p < num_planes_; ++p) {
                double dot = v.dot(hyperplanes_[t][p]);
                bits.push_back(dot >= 0.0 ? '1' : '0');
            }

            hashes.push_back(bits);
        }

        return hashes;
    }
};
