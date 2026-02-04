// tinyllama_cuda inference binary.
//
// Usage:
//   inference --weights weights/tinyllama-int8.bin --tokens "1 15043 590" \
//             --max-tokens 128 [--bench]
//
// Prints one generated token id per line on stdout. Timing info goes to stderr.
// Token encoding/decoding is handled by scripts/run.py — this keeps the C++
// side free of any tokenizer dependency.

#include "kernels.hpp"
#include "model.hpp"
#include "weights.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string weights_path;
    std::string tokens_str;
    int         max_tokens = 128;
    bool        bench      = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if (k == "--weights")         a.weights_path = next();
        else if (k == "--tokens")     a.tokens_str   = next();
        else if (k == "--max-tokens") a.max_tokens   = std::stoi(next());
        else if (k == "--bench")      a.bench        = true;
        else if (k == "-h" || k == "--help") {
            std::fprintf(stderr,
                "usage: %s --weights PATH --tokens \"id id id\" [--max-tokens N] [--bench]\n",
                argv[0]);
            std::exit(0);
        }
        else throw std::runtime_error("unknown arg: " + k);
    }
    if (a.weights_path.empty()) throw std::runtime_error("--weights is required");
    if (a.tokens_str.empty())   throw std::runtime_error("--tokens is required");
    return a;
}

std::vector<int32_t> parse_tokens(const std::string& s) {
    std::vector<int32_t> out;
    std::istringstream is(s);
    int32_t t;
    while (is >> t) out.push_back(t);
    if (out.empty()) throw std::runtime_error("no input tokens");
    return out;
}

}  // namespace

int main(int argc, char** argv) try {
    Args args = parse_args(argc, argv);
    std::vector<int32_t> prompt = parse_tokens(args.tokens_str);

    std::fprintf(stderr, "Loading weights: %s\n", args.weights_path.c_str());
    tllm::Weights weights(args.weights_path);
    const auto& cfg = weights.config();
    std::fprintf(stderr, "  V=%u D=%u I=%u L=%u H=%u Hkv=%u Dh=%u\n",
                 cfg.vocab_size, cfg.hidden_dim, cfg.intermediate_dim,
                 cfg.num_layers, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);

    tllm::ModelState state(cfg);

    const int prompt_len = int(prompt.size());
    if (prompt_len + args.max_tokens > int(cfg.max_seq_len)) {
        throw std::runtime_error("prompt + max_tokens exceeds max_seq_len");
    }

    // --- Prefill ---
    int32_t last = 0;
    auto t_prefill_start = std::chrono::steady_clock::now();
    for (int i = 0; i < prompt_len; ++i) {
        last = state.forward(weights, prompt[i], i);
    }
    cudaDeviceSynchronize();
    auto t_prefill_end = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(
        t_prefill_end - t_prefill_start).count();

    // `last` now holds the greedy prediction for the token immediately after
    // the prompt. Emit it, then continue decoding.
    std::printf("%d\n", last);

    // --- Decode ---
    auto t_decode_start = std::chrono::steady_clock::now();
    int generated = 1;
    for (; generated < args.max_tokens; ++generated) {
        int32_t pos = prompt_len + generated - 1;
        last = state.forward(weights, last, pos);
        std::printf("%d\n", last);
    }
    cudaDeviceSynchronize();
    auto t_decode_end = std::chrono::steady_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(
        t_decode_end - t_decode_start).count();

    std::fprintf(stderr, "prefill: %.2f ms (%d tokens)\n", prefill_ms, prompt_len);
    std::fprintf(stderr, "decode:  %.2f ms (%d tokens)\n", decode_ms, generated);
    if (args.bench && generated > 1) {
        double tps = (generated - 1) * 1000.0 / (decode_ms);
        std::fprintf(stderr, "decode tokens/sec: %.2f\n", tps);
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
