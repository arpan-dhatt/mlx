// Copyright © 2023 Apple Inc.

#include <chrono>
#include <cmath>
#include <iostream>

#include "mlx/mlx.h"
#include "timer.h"

/**
 * An example of linear regression with MLX.
 */
using namespace mlx::core;

class FeedForward {
public:
    explicit FeedForward(int dim, int hidden_dim) : w1(mlx::core::random::normal({dim, hidden_dim}, mlx::core::float16, 0.0, 0.0025)), w2(mlx::core::random::normal({hidden_dim, dim}, mlx::core::float16, 0.0, 0.0025)), w3(mlx::core::random::normal({dim, hidden_dim}, mlx::core::float16, 0.0, 0.0025)) {
        mlx::core::eval({w1, w2, w3});
    }
    
    mlx::core::array forward(const mlx::core::array& input) {
        auto x = mlx::core::matmul(input, w1);
        auto act = x * mlx::core::sigmoid(x) * mlx::core::matmul(input, w3);
        return mlx::core::matmul(act, w2);
    }
    
private:
    mlx::core::array w1, w2, w3;
};

class QuantizedFeedForward {
public:
    explicit QuantizedFeedForward(int dim, int hidden_dim) {
        auto w1 = mlx::core::random::normal({hidden_dim, dim}, mlx::core::float16, 0.0, 0.0025);
        auto w2 = mlx::core::random::normal({dim, hidden_dim}, mlx::core::float16, 0.0, 0.0025);
        auto w3 = mlx::core::random::normal({hidden_dim, dim}, mlx::core::float16, 0.0, 0.0025);
        
        auto [w1_q, w1_s, w1_b] = mlx::core::quantize(w1);
        auto [w2_q, w2_s, w2_b] = mlx::core::quantize(w2);
        auto [w3_q, w3_s, w3_b] = mlx::core::quantize(w3);
        
        layers.push_back(w1_q);
        layers.push_back(w1_s);
        layers.push_back(w1_b);
        
        layers.push_back(w2_q);
        layers.push_back(w2_s);
        layers.push_back(w2_b);
        
        layers.push_back(w3_q);
        layers.push_back(w3_s);
        layers.push_back(w3_b);
        
        mlx::core::eval(layers);
    }
    
    mlx::core::array forward(const mlx::core::array& input) {
        auto x = mlx::core::quantized_matmul(input, layers[0], layers[1], layers[2]);
        auto act = x * mlx::core::sigmoid(x) * mlx::core::quantized_matmul(input, layers[6], layers[7], layers[8]);
        return mlx::core::quantized_matmul(act, layers[3], layers[4], layers[5]);
    }
    
private:
    std::vector<mlx::core::array> layers;
};

int main() {
    int BATCH_SIZE = 1;
    int HEADS = 128;
    auto input = mlx::core::random::normal({256, 4096}, mlx::core::float16, 0.0, 0.1);
    mlx::core::eval({input});
    
    auto ff = FeedForward(4096, 14336);
    auto ffout = ff.forward(input);
    
    auto qff = QuantizedFeedForward(4096, 14336);
    auto qffout = qff.forward(input);
    
    metal::start_capture("/Users/arpandhatt/Downloads/ff.gputrace");
    mlx::core::eval({ffout});
    metal::stop_capture();
        
    metal::start_capture("/Users/arpandhatt/Downloads/qff.gputrace");
    mlx::core::eval({qffout});
    metal::stop_capture();
    
    auto diff = qffout - ffout;
    std::cout << mlx::core::mean(diff).item<float>() << " " << mlx::core::max(mlx::core::abs(diff)).item<float>() << std::endl;
}
