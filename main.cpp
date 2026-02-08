#include <vector>
#include <string>
#include <cmath>
#include <set>
#include <random>
#include <fstream>
#include <cstdint>
#include <algorithm>

class HashFuncGen {
public:
    static uint32_t hash(const std::string& key) {
        const uint32_t FNV_PRIME = 16777619u;
        const uint32_t FNV_OFFSET_BASIS = 2166136261u;
        uint32_t h = FNV_OFFSET_BASIS;
        for (unsigned char c : key) {
            h ^= c;
            h *= FNV_PRIME;
        }
        return h;
    }
};

class RandomStreamGen {
    std::mt19937 gen;
    const std::string chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-";
    std::uniform_int_distribution<> len_dist;
    std::uniform_int_distribution<> char_dist;
public:
    RandomStreamGen() : gen(42), len_dist(1, 30), char_dist(0, 62) {}
    void set_seed(int seed) {
        gen.seed(static_cast<unsigned int>(seed));
    }
    std::string next() {
        int len = len_dist(gen);
        std::string res;
        res.reserve(len);
        for (int i = 0; i < len; i++) {
            res += chars[char_dist(gen)];
        }
        return res;
    }
};

class HyperLogLog {
    int b;
    int m;
    double alpha;
    std::vector<uint8_t> registers;

    double get_alpha(int q) {
        if (q == 16) {
            return 0.673;
        }
        if (q == 32) {
            return 0.697;
        }
        if (q == 64) {
            return 0.709;
        }
        return 0.7213 / (1.0 + 1.079 / static_cast<double>(q));
    }

    uint8_t get_rank(uint32_t hash_val) {
        uint32_t w = hash_val << b;
        if (w == 0) return static_cast<uint8_t>((32 - b) + 1);
        return static_cast<uint8_t>(__builtin_clz(w) + 1);
    }

public:
    HyperLogLog(int b_bits) : b(b_bits) {
        m = 1 << b;
        registers.assign(m, 0);
        alpha = get_alpha(m);
    }

    void add(const std::string& str) {
        uint32_t x = HashFuncGen::hash(str);
        uint32_t j = x >> (32 - b);
        uint8_t rho = get_rank(x);
        if (rho > registers[j]) {
            registers[j] = rho;
        }
    }

    double estimate() const {
        double sum_inv = 0.0;
        int V = 0;
        for (uint8_t val : registers) {
            sum_inv += std::pow(2.0, -static_cast<int>(val));
            if (val == 0) V++;
        }

        double E = (alpha * static_cast<double>(m) * static_cast<double>(m)) / sum_inv;

        if (E <= 2.5 * m) {
            if (V > 0) {
                E = static_cast<double>(m) * std::log(static_cast<double>(m) / static_cast<double>(V));
            }
        }
        return E;
    }
};

struct ExperimentResult {
    double true_val;
    double estimate;
};

int main() {
    int stream_size = 100000;
    int step_size = 2000;
    int steps = 50;
    int runs = 20;
    int b = 12;

    std::ofstream file("hll_results.csv");
    file << "step_num,mean_true,mean_est,std_dev_est,theory_error_bound\n";

    std::vector<std::vector<ExperimentResult>> exp(steps);

    for (int ex = 0; ex < runs; ex++) {
        RandomStreamGen gen;
        gen.set_seed(ex + 1237);

        HyperLogLog hll(b);
        std::set<std::string> exact_counter;

        for (int s = 0; s < steps; s++) {
            for (int k = 0; k < step_size; k++) {
                std::string str = gen.next();
                hll.add(str);
                exact_counter.insert(str);
            }
            exp[s].push_back({static_cast<double>(exact_counter.size()), hll.estimate()});
        }
    }

    double theory_error = 1.04 / std::sqrt(1 << b);

    for (int s = 0; s < steps; s++) {
        double sum_true = 0;
        double sum_est = 0;

        for (const auto& res : exp[s]) {
            sum_true += res.true_val;
            sum_est += res.estimate;
        }

        double mean_true = sum_true / runs;
        double mean_est = sum_est / runs;

        double sq_sum = 0;
        for (const auto& res : exp[s]) {
            sq_sum += (res.estimate - mean_est) * (res.estimate - mean_est);
        }
        double std_dev = std::sqrt(sq_sum / runs);
        double theory_abs_error = mean_true * theory_error;

        file << (s + 1) * step_size << ","
             << mean_true << ","
             << mean_est << ","
             << std_dev << ","
             << theory_abs_error << "\n";
    }
    file.close();
    return 0;
}