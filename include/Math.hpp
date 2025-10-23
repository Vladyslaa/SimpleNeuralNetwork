#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <execution>
#include <numeric>
#include <cmath>

/**
 * Math.hpp
 * A Custom library for common neural network functions and linear algebra utilities.
 * It's a bit heavy for this project,
 * but for larger NN it becomes more efficient.
 */
namespace math {

#pragma region linear_algebra

    const double EPS = 1e-12;

    inline std::vector<double> operator+(const std::vector<double>& vec1, const std::vector<double>& vec2) {
        if (vec1.size() != vec2.size()) throw std::invalid_argument("Vectors must have the same size");

        std::vector<double> res(vec1.size());
        std::transform(
            std::execution::par_unseq, 
            vec1.begin(), vec1.end(), 
            vec2.begin(), res.begin(), 
            [](double a, double b) { return a + b; }
        );
        
        return res;
    }

    inline std::vector<double> operator-(const std::vector<double>& vec1, const std::vector<double>& vec2) {
        if (vec1.size() != vec2.size()) throw std::invalid_argument("Vectors must have the same size");

        std::vector<double> res(vec1.size());
        std::transform(
            std::execution::par_unseq, 
            vec1.begin(), vec1.end(), 
            vec2.begin(), res.begin(), 
            [](double a, double b) { return a - b; }
        );

        return res;
    }

    inline std::vector<double> operator*(const std::vector<double>& vec, const double& num) {
        std::vector<double> res(vec.size());
        std::transform(
            std::execution::par_unseq,
            vec.begin(), vec.end(), 
            res.begin(), [num](double a) { return a * num; }
        );
        
        return res;
    }

    inline double operator*(const std::vector<double>& vec1, const std::vector<double>& vec2) {
        if (vec1.size() != vec2.size()) throw std::invalid_argument("Vectors must have the same size");
        return std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0);
    }

    inline std::vector<std::vector<double>> operator+(const std::vector<std::vector<double>>& mtx1, const std::vector<std::vector<double>>& mtx2) {
        if (mtx1.size() != mtx2.size()) throw std::invalid_argument("Matrixs must have the same size");

        std::vector<std::vector<double>> res(mtx1.size(), std::vector<double>(mtx1[0].size()));
        std::transform(
            std::execution::par_unseq, 
            mtx1.begin(), mtx1.end(), 
            mtx2.begin(), res.begin(), 
            [](const std::vector<double>& a, const std::vector<double>& b) { return a + b; }
        );

        return res;
    }

    inline std::vector<std::vector<double>> operator-(const std::vector<std::vector<double>>& mtx1, const std::vector<std::vector<double>>& mtx2) {
        if (mtx1.size() != mtx2.size()) throw std::invalid_argument("Matrixs must have the same size");

        std::vector<std::vector<double>> res(mtx1.size(), std::vector<double>(mtx1[0].size()));
        std::transform(
            std::execution::par_unseq, 
            mtx1.begin(), mtx1.end(), 
            mtx2.begin(), res.begin(), 
            [](const std::vector<double>& a, const std::vector<double>& b) { return a - b; }
        );

        return res;
    }

    inline std::vector<std::vector<double>> operator*(const std::vector<std::vector<double>>& mtx, double num) {
        std::vector<std::vector<double>> res(mtx.size(), std::vector<double>(mtx[0].size()));
        std::transform(
            std::execution::par_unseq,
            mtx.begin(), mtx.end(),
            res.begin(),
            [num](const std::vector<double>& a) { return a * num; }
        );

        return res;
    }

    inline std::vector<double> operator*(const std::vector<std::vector<double>>& mtx, const std::vector<double>& vec) {
        if (mtx[0].size() != vec.size()) throw std::invalid_argument("Matrix row and vector must have the same size");
        size_t size = mtx.size();

        std::vector<double> res(size);
        for (size_t i = 0; i < size; ++i) {
            res[i] = std::inner_product(mtx[i].begin(), mtx[i].end(), vec.begin(), 0.0);
        }

        return res;
    }

#pragma endregion
#pragma region neural_network

    inline double sigmoid(const double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline double sigmoid_derivative(const double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }

    inline double relu(const double x) {
        return std::max(0.0, x);
    }

    inline double relu_derivative(const double x) {
        return x > 0.0 ? 1.0 : 0.0;
    }

    inline double tanh(const double x) { return std::tanh(x); }

    inline double tanh_derivative(const double x) {
        double t = std::tanh(x);
        return 1.0 - t * t;
    }

    inline double bce(const double answ, double pred) {
        pred = std::clamp(pred, EPS, 1.0 - EPS);
        return -(answ * std::log(pred) + (1.0 - answ) * std::log(1.0 - pred));
    }

    inline double bce_with_logits_loss(const double logit, const double answ) {
        return std::max(logit, 0.0) - logit * answ + std::log(1 + std::exp(-std::abs(logit)));
    }

    inline double bce_with_logits_loss_delta(const double logit, const double answ) {
        return sigmoid(logit) - answ;
    }

    inline double bce_delta(const double answ, const double pred) {
        return pred - answ;
    }

    inline std::vector<std::vector<double>> weights_gradient(std::vector<double> delt, std::vector<double> inpt) {
        std::vector<std::vector<double>> res(delt.size(), std::vector<double>(inpt.size()));

        for (size_t i = 0; i < delt.size(); ++i) {
            for (size_t j = 0; j < inpt.size(); ++j) {
                res[i][j] = delt[i] * inpt[j];
            }
        }

        return res;
    }

    inline double xavier_limit(const double in, const double out) {
        return std::sqrt(6.0 / (in + out));
    }

#pragma endregion

}