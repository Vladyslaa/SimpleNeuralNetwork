#pragma once

#include <array>
#include <random>
#include <stdexcept>
#include <vector>
/**
 * Random.hpp
 * My custom header-only library for random number generation.
 * In this project, only Random::Double() is used.
 */
class Random {
private:
    static inline int seed = 0;
    static inline bool isInitialized = false;
    static inline std::mt19937 engine;

    Random() = delete;
    ~Random() = delete;

public:
    /**
     * Initialize an engine
     * @attention call this method first before calling any other method of this class
     */
    static void Init(int initSeed) {
        if (!isInitialized) {
            seed = initSeed;
            engine.seed(initSeed);
            isInitialized = true;
        }
    }

    /**
     * @return random float in range [min ; max]
     */
    static double Double(double min, double max) {
        if (!isInitialized) {
            throw std::logic_error("Random generator isn't initialized! Call Random::Init() first");
        }

        std::uniform_real_distribution<double> dist(min, max);
        return dist(engine);
    }

#ifdef FULL_RANDOM
    /**
     * @return randoms engine
     */
    static std::mt19937 Engine() {
        if (!isInitialized) {
            throw std::logic_error("Random generator isn't initialized! Call Random::Init() first");
        }

        return engine;
    }

    /**
     * @return random int in range min - max
     */
    static int Int(int min, int max) {
        if (!isInitialized) {
            throw std::logic_error("Random generator isn't initialized! Call Random::Init() first");
        }

        std::uniform_int_distribution<int> dist(min, max);
        return dist(engine);
    }

    /**
     * @return random float in range min - max
     */
    static float Float(float min, float max) {
        if (!isInitialized) {
            throw std::logic_error("Random generator isn't initialized! Call Random::Init() first");
        }

        std::uniform_real_distribution<float> dist(min, max);
        return dist(engine);
    }


    /**
     * Uses a uniform random distribution
     * @return true with the specified chance percentage
     * @param p chance (0 - 100)
     */
    static bool Chance(float p) {
        if (!isInitialized) {
            throw std::logic_error("Random generator isn't initialized! Call Random::Init() first");
        }

        std::uniform_real_distribution<float> dist(0.0f, 100.0f);
        return dist(engine) <= p;
    }

    /**
     * @return random element from vector
     */
    template <typename T>
    static T Element(const std::vector<T>& vector) {
        if (!isInitialized) {
            throw std::logic_error("Random generator isn't initialized! Call Random::Init() first");
        }

        if (vector.empty()) {
            throw std::out_of_range("Cannot get random element from empty vector");
        }

        return vector[Int(0, vector.size() - 1)];
    }

    /**
     * @return random element from array
     */
    template <typename T, std::size_t N>
    static T Element(const std::array<T, N>& array) {
        if (!isInitialized) {
            throw std::logic_error("Random generator isn't initialized! Call Random::Init() first");
        }

        if (array.empty()) {
            throw std::out_of_range("Cannot get random element from empty array");
        }

        return array[Int(0, N - 1)];
    }

    /**
     * Generates a random value within ±p% range of the given base value
     *
     * @tparam T arithmetic type (int, float, double and their variants)
     * @param var base value around which the random number is generated
     * @param p percentage deviation from the base value (e.g., 10.0f for ±10%)
     * @return random value in range [var - p%, var + p%]
     */
    template <typename T>
    static T Average(T var, float p) {
        if (!isInitialized) {
            throw std::logic_error("Random generator isn't initialized! Call Random::Init() first");
        }

        static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, "T must be arithmetic but not bool");

        double res = Double(var * (1.0 - p / 100.0), var * (1.0 + p / 100.0));

        if constexpr (std::is_integral_v<T>) {
            return Mathf::RoundToInt(res);
        } else {
            if constexpr (std::is_same_v<T, float>) {
                return static_cast<float>(res);
            } else if constexpr (std::is_same_v<T, double>) {
                return res;
            }
        }

        throw std::runtime_error("something happened");
    }
#endif
};