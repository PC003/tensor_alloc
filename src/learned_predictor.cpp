#include "learned_predictor.h"
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <mutex>
#include <memory>

struct CppTreeNode {
    bool is_leaf;
    int prediction;
    int feature_idx;
    double threshold;
    int left_idx;
    int right_idx;
};

class CppDecisionTreePredictor : public IPredictor {
public:
    std::vector<CppTreeNode> nodes;

    int predict_class(const tensor_alloc_request_t* req) override {
        if (!req || nodes.empty()) {
            return TENSOR_SIZE_CLASS_INVALID;
        }

        // Extract 11 causal features
        double features[11];
        features[0] = static_cast<double>(req->ndim);
        features[1] = (req->ndim > 0 && req->shape) ? static_cast<double>(req->shape[0]) : 1.0;
        features[2] = (req->ndim > 1 && req->shape) ? static_cast<double>(req->shape[1]) : 1.0;
        features[3] = (req->ndim > 2 && req->shape) ? static_cast<double>(req->shape[2]) : 1.0;
        features[4] = (req->ndim > 3 && req->shape) ? static_cast<double>(req->shape[3]) : 1.0;

        double num_elements = 1.0;
        if (req->ndim > 0 && req->shape) {
            for (size_t i = 0; i < req->ndim; ++i) {
                num_elements *= static_cast<double>(req->shape[i]);
            }
        }
        features[5] = num_elements;
        features[6] = (req->strides == nullptr) ? 1.0 : 0.0;
        features[7] = static_cast<double>(req->dtype);
        features[8] = static_cast<double>(req->lifetime_hint);
        features[9] = static_cast<double>(req->requested_bytes);
        features[10] = (req->requested_bytes > 0) ? std::log2(static_cast<double>(req->requested_bytes)) : 0.0;

        // Tree traversal
        int curr = 0;
        int max_steps = 64; // Guard against cyclic corruption
        while (curr >= 0 && curr < static_cast<int>(nodes.size()) && max_steps-- > 0) {
            const CppTreeNode& n = nodes[curr];
            if (n.is_leaf) {
                return n.prediction;
            }
            if (n.feature_idx < 0 || n.feature_idx >= 11) {
                return TENSOR_SIZE_CLASS_INVALID;
            }
            if (features[n.feature_idx] <= n.threshold) {
                curr = n.left_idx;
            } else {
                curr = n.right_idx;
            }
        }

        return TENSOR_SIZE_CLASS_INVALID;
    }
};

static std::mutex g_model_mutex;
static std::unique_ptr<CppDecisionTreePredictor> g_loaded_predictor;

// Lightweight recursive JSON tree parser
static int parse_json_node(const std::string& json, size_t& pos, std::vector<CppTreeNode>& out_nodes) {
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\r' || json[pos] == '\n')) pos++;
    if (pos >= json.size() || json[pos] != '{') return -1;

    size_t node_idx = out_nodes.size();
    out_nodes.emplace_back();
    CppTreeNode& n = out_nodes[node_idx];
    n.is_leaf = false;
    n.prediction = 0;
    n.feature_idx = -1;
    n.threshold = 0.0;
    n.left_idx = -1;
    n.right_idx = -1;

    pos++; // Skip '{'

    while (pos < json.size() && json[pos] != '}') {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\r' || json[pos] == '\n' || json[pos] == ',')) pos++;
        if (pos >= json.size() || json[pos] == '}') break;

        // Read key
        if (json[pos] != '"') { pos++; continue; }
        pos++;
        size_t key_end = json.find('"', pos);
        if (key_end == std::string::npos) break;
        std::string key = json.substr(pos, key_end - pos);
        pos = key_end + 1;

        // Skip to value
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':')) pos++;

        if (key == "is_leaf") {
            if (json.substr(pos, 4) == "true") {
                out_nodes[node_idx].is_leaf = true;
                pos += 4;
            } else {
                out_nodes[node_idx].is_leaf = false;
                pos += 5;
            }
        } else if (key == "prediction") {
            size_t val_end = pos;
            while (val_end < json.size() && (std::isdigit(json[val_end]) || json[val_end] == '-')) val_end++;
            out_nodes[node_idx].prediction = std::stoi(json.substr(pos, val_end - pos));
            pos = val_end;
        } else if (key == "feature_idx") {
            size_t val_end = pos;
            while (val_end < json.size() && std::isdigit(json[val_end])) val_end++;
            out_nodes[node_idx].feature_idx = std::stoi(json.substr(pos, val_end - pos));
            pos = val_end;
        } else if (key == "threshold") {
            size_t val_end = pos;
            while (val_end < json.size() && (std::isdigit(json[val_end]) || json[val_end] == '.' || json[val_end] == '-')) val_end++;
            out_nodes[node_idx].threshold = std::stod(json.substr(pos, val_end - pos));
            pos = val_end;
        } else if (key == "left") {
            int left_child = parse_json_node(json, pos, out_nodes);
            out_nodes[node_idx].left_idx = left_child;
        } else if (key == "right") {
            int right_child = parse_json_node(json, pos, out_nodes);
            out_nodes[node_idx].right_idx = right_child;
        } else {
            // Skip other fields (samples, impurity, feature_name)
            if (json[pos] == '"') {
                pos++;
                size_t str_end = json.find('"', pos);
                pos = (str_end != std::string::npos) ? str_end + 1 : json.size();
            } else {
                while (pos < json.size() && json[pos] != ',' && json[pos] != '}') pos++;
            }
        }
    }

    if (pos < json.size() && json[pos] == '}') pos++;
    return static_cast<int>(node_idx);
}

tensor_status_t learned_predictor_load(const char* json_filepath) {
    if (!json_filepath) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    std::ifstream file(json_filepath);
    if (!file.is_open()) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Locate "tree": {
    size_t tree_pos = content.find("\"tree\"");
    if (tree_pos == std::string::npos) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }
    tree_pos = content.find('{', tree_pos);
    if (tree_pos == std::string::npos) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    auto predictor = std::make_unique<CppDecisionTreePredictor>();
    int root = parse_json_node(content, tree_pos, predictor->nodes);

    if (root != 0 || predictor->nodes.empty()) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    {
        std::lock_guard<std::mutex> lock(g_model_mutex);
        g_loaded_predictor = std::move(predictor);
    }

    return TENSOR_STATUS_SUCCESS;
}

IPredictor* learned_predictor_get(void) {
    std::lock_guard<std::mutex> lock(g_model_mutex);
    return g_loaded_predictor.get();
}

void learned_predictor_destroy(void) {
    std::lock_guard<std::mutex> lock(g_model_mutex);
    g_loaded_predictor.reset();
}

bool learned_predictor_is_loaded(void) {
    std::lock_guard<std::mutex> lock(g_model_mutex);
    return (g_loaded_predictor != nullptr);
}
