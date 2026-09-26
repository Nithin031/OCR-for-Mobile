// gemma_runner: a line-oriented JSON front end to the pinned llama.cpp C API.
//
// The G0 pipeline drives it over stdin/stdout. It uses the same calls the
// Android JNI layer (G1) will use: llama_chat_apply_template with the chat
// template stored in the GGUF, llama_tokenize(add_special=true,
// parse_special=true), prompt decode in n_batch chunks, then greedy sampling
// until an end-of-generation token or n_predict.
//
// Requests (one JSON object per line):
//   {"op":"render",   "messages":[{"role":..,"content":..},...], "add_assistant":true}
//   {"op":"tokenize", "text":"...", "add_special":true, "parse_special":true}
//   {"op":"count",    "messages":[...]}
//   {"op":"generate", "messages":[...], "n_predict":512}
//   {"op":"quit"}
// Every response is one JSON line with "ok": true|false. llama.cpp logs go to stderr.

#include "ggml-backend.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using json = nlohmann::ordered_json;

namespace {

struct Options {
    std::string model;
    int n_ctx = 4096;
    int n_batch = 2048;
    int n_ubatch = 512;
    int n_threads = 4;
    int n_threads_batch = 4;
    bool vocab_only = false;
    std::string flash_attn = "disabled";
    bool swa_full = true;
    std::string load_mode = "mmap";
    std::string kv_type = "f16";
    bool extra_bufts = true;
};

[[noreturn]] void usage(const char * msg) {
    std::fprintf(stderr,
        "error: %s\n"
        "usage: gemma_runner --model FILE [--vocab-only] [--n-ctx N] [--n-batch N] [--n-ubatch N]\n"
        "                    [--threads N] [--threads-batch N] [--flash-attn disabled|enabled|auto]\n"
        "                    [--swa-full 0|1] [--load-mode mmap|none] [--kv-type f16] [--extra-bufts 0|1]\n", msg);
    std::exit(2);
}

Options parse_args(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage(("missing value for " + a).c_str());
            return argv[++i];
        };
        if      (a == "--model")         o.model = next();
        else if (a == "--vocab-only")    o.vocab_only = true;
        else if (a == "--n-ctx")         o.n_ctx = std::stoi(next());
        else if (a == "--n-batch")       o.n_batch = std::stoi(next());
        else if (a == "--n-ubatch")      o.n_ubatch = std::stoi(next());
        else if (a == "--threads")       o.n_threads = std::stoi(next());
        else if (a == "--threads-batch") o.n_threads_batch = std::stoi(next());
        else if (a == "--flash-attn")    o.flash_attn = next();
        else if (a == "--swa-full")      o.swa_full = next() != "0";
        else if (a == "--load-mode")     o.load_mode = next();
        else if (a == "--kv-type")       o.kv_type = next();
        else if (a == "--extra-bufts")   o.extra_bufts = next() != "0";
        else usage(("unknown argument " + a).c_str());
    }
    if (o.model.empty()) usage("--model is required");
    if (o.kv_type != "f16") usage("only --kv-type f16 is supported (spec: KV cache F16)");
    return o;
}

llama_flash_attn_type flash_attn_type(const std::string & s) {
    if (s == "disabled") return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (s == "enabled")  return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (s == "auto")     return LLAMA_FLASH_ATTN_TYPE_AUTO;
    usage(("bad --flash-attn " + s).c_str());
}

llama_load_mode load_mode(const std::string & s) {
    if (s == "mmap") return LLAMA_LOAD_MODE_MMAP;
    if (s == "none") return LLAMA_LOAD_MODE_NONE;
    usage(("bad --load-mode " + s).c_str());
}

double ms_since(int64_t t0_us) { return (llama_time_us() - t0_us) / 1000.0; }

std::vector<std::pair<std::string, std::string>> read_messages(const json & req) {
    if (!req.contains("messages") || !req["messages"].is_array()) {
        throw std::runtime_error("request needs a 'messages' array");
    }
    std::vector<std::pair<std::string, std::string>> out;
    for (const auto & m : req["messages"]) {
        out.emplace_back(m.at("role").get<std::string>(), m.at("content").get<std::string>());
    }
    return out;
}

std::string apply_template(const char * tmpl, const std::vector<std::pair<std::string, std::string>> & msgs, bool add_ass) {
    std::vector<llama_chat_message> chat;
    size_t total = 0;
    for (const auto & m : msgs) {
        chat.push_back({m.first.c_str(), m.second.c_str()});
        total += m.first.size() + m.second.size();
    }
    std::vector<char> buf(2 * total + 1024);
    int32_t n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_ass, buf.data(), (int32_t) buf.size());
    if (n < 0) {
        throw std::runtime_error("llama_chat_apply_template does not support this model's chat template");
    }
    if ((size_t) n > buf.size()) {
        buf.resize(n);
        n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_ass, buf.data(), (int32_t) buf.size());
    }
    return std::string(buf.data(), n);
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special, bool parse_special) {
    int32_t n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, add_special, parse_special);
    if (n == INT32_MIN) throw std::runtime_error("tokenization overflow");
    if (n < 0) n = -n;
    std::vector<llama_token> tokens(n);
    int32_t r = llama_tokenize(vocab, text.data(), (int32_t) text.size(), tokens.data(), n, add_special, parse_special);
    if (r < 0) throw std::runtime_error("tokenization failed");
    tokens.resize(r);
    return tokens;
}

std::string detokenize(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string out(std::max<size_t>(64, tokens.size() * 16), '\0');
    int32_t n = llama_detokenize(vocab, tokens.data(), (int32_t) tokens.size(), out.data(), (int32_t) out.size(),
                                 /*remove_special=*/ false, /*unparse_special=*/ false);
    if (n < 0) {
        out.resize(-n);
        n = llama_detokenize(vocab, tokens.data(), (int32_t) tokens.size(), out.data(), (int32_t) out.size(), false, false);
        if (n < 0) throw std::runtime_error("detokenization failed");
    }
    out.resize(n);
    return out;
}

json generate(llama_context * ctx, const llama_vocab * vocab, const char * tmpl, const json & req, int n_batch) {
    const auto msgs = read_messages(req);
    const int n_predict = req.value("n_predict", 0);
    if (n_predict <= 0) throw std::runtime_error("n_predict must be > 0");

    const std::string rendered = apply_template(tmpl, msgs, true);
    std::vector<llama_token> prompt = tokenize(vocab, rendered, true, true);
    const int n_ctx = (int) llama_n_ctx(ctx);
    if ((int) prompt.size() + n_predict > n_ctx) {
        throw std::runtime_error("prompt (" + std::to_string(prompt.size()) + " tokens) + n_predict (" +
                                 std::to_string(n_predict) + ") exceeds n_ctx (" + std::to_string(n_ctx) + ")");
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_perf_context_reset(ctx);

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    const int64_t t_start = llama_time_us();
    for (size_t i = 0; i < prompt.size(); i += n_batch) {
        const int32_t n = (int32_t) std::min<size_t>(n_batch, prompt.size() - i);
        const int rc = llama_decode(ctx, llama_batch_get_one(prompt.data() + i, n));
        if (rc != 0) {
            llama_sampler_free(smpl);
            throw std::runtime_error("llama_decode failed on the prompt, code " + std::to_string(rc));
        }
    }
    const double prompt_ms = ms_since(t_start);

    std::vector<llama_token> generated;
    std::string stop_reason = "length";
    llama_token eog = -1;
    double ttft_ms = 0.0;
    int n_decode_calls = 0;
    const int64_t t_gen = llama_time_us();
    for (int step = 0; step < n_predict; ++step) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (step == 0) ttft_ms = ms_since(t_start);
        if (llama_vocab_is_eog(vocab, tok)) {
            stop_reason = "eog";
            eog = tok;
            break;
        }
        generated.push_back(tok);
        if (step + 1 == n_predict) break;
        const int rc = llama_decode(ctx, llama_batch_get_one(&tok, 1));
        if (rc != 0) {
            llama_sampler_free(smpl);
            throw std::runtime_error("llama_decode failed while generating, code " + std::to_string(rc));
        }
        ++n_decode_calls;
    }
    const double gen_ms = ms_since(t_gen);
    llama_sampler_free(smpl);

    const llama_perf_context_data perf = llama_perf_context(ctx);
    json r;
    r["ok"] = true;
    r["rendered"] = rendered;
    r["prompt_tokens"] = prompt;
    r["n_prompt_tokens"] = prompt.size();
    r["tokens"] = generated;
    r["n_tokens"] = generated.size();
    r["text"] = detokenize(vocab, generated);
    r["stop_reason"] = stop_reason;
    r["eog_token"] = eog;
    r["timings"] = {
        {"prompt_ms", prompt_ms},
        {"prompt_tokens_per_s", prompt_ms > 0 ? prompt.size() * 1000.0 / prompt_ms : 0.0},
        {"ttft_ms", ttft_ms},
        {"generation_ms", gen_ms},
        {"decode_calls", n_decode_calls},
        {"decode_tokens_per_s", gen_ms > 0 && n_decode_calls > 0 ? n_decode_calls * 1000.0 / gen_ms : 0.0},
        {"perf_t_p_eval_ms", perf.t_p_eval_ms},
        {"perf_n_p_eval", perf.n_p_eval},
        {"perf_t_eval_ms", perf.t_eval_ms},
        {"perf_n_eval", perf.n_eval},
    };
    return r;
}

void reply(const json & j) {
    std::cout << j.dump(-1, ' ', /*ensure_ascii=*/ true, json::error_handler_t::replace) << "\n" << std::flush;
}

}  // namespace

int main(int argc, char ** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    const Options opt = parse_args(argc, argv);

    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.load_mode = load_mode(opt.load_mode);
    mparams.vocab_only = opt.vocab_only;
    mparams.use_extra_bufts = opt.extra_bufts;

    const int64_t t_load = llama_time_us();
    llama_model * model = llama_model_load_from_file(opt.model.c_str(), mparams);
    if (model == nullptr) {
        reply({{"ok", false}, {"event", "ready"}, {"error", "failed to load model " + opt.model}});
        return 1;
    }
    const double load_model_ms = ms_since(t_load);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * tmpl = llama_model_chat_template(model, /*name=*/ nullptr);
    if (tmpl == nullptr) {
        reply({{"ok", false}, {"event", "ready"}, {"error", "the GGUF has no tokenizer.chat_template"}});
        llama_model_free(model);
        return 1;
    }

    llama_context * ctx = nullptr;
    double init_context_ms = 0.0;
    if (!opt.vocab_only) {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = opt.n_ctx;
        cparams.n_batch = opt.n_batch;
        cparams.n_ubatch = opt.n_ubatch;
        cparams.n_threads = opt.n_threads;
        cparams.n_threads_batch = opt.n_threads_batch;
        cparams.flash_attn_type = flash_attn_type(opt.flash_attn);
        cparams.type_k = GGML_TYPE_F16;
        cparams.type_v = GGML_TYPE_F16;
        cparams.swa_full = opt.swa_full;
        cparams.no_perf = false;
        const int64_t t_ctx = llama_time_us();
        ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr) {
            reply({{"ok", false}, {"event", "ready"}, {"error", "failed to create the llama context"}});
            llama_model_free(model);
            return 1;
        }
        init_context_ms = ms_since(t_ctx);
    }

    char desc[256] = {0};
    llama_model_desc(model, desc, sizeof(desc));
    json ready;
    ready["ok"] = true;
    ready["event"] = "ready";
    ready["load_model_ms"] = load_model_ms;
    ready["init_context_ms"] = init_context_ms;
    ready["vocab_only"] = opt.vocab_only;
    ready["n_ctx"] = ctx ? (int) llama_n_ctx(ctx) : 0;
    ready["model_desc"] = desc;
    ready["model_size_bytes"] = llama_model_size(model);
    ready["model_n_params"] = llama_model_n_params(model);
    ready["vocab_n_tokens"] = llama_vocab_n_tokens(vocab);
    ready["add_bos"] = llama_vocab_get_add_bos(vocab);
    ready["bos_token"] = llama_vocab_bos(vocab);
    ready["chat_template"] = tmpl;
    ready["system_info"] = llama_print_system_info();
    reply(ready);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        try {
            const json req = json::parse(line);
            const std::string op = req.at("op").get<std::string>();
            if (op == "quit") {
                reply({{"ok", true}, {"op", "quit"}});
                break;
            } else if (op == "render") {
                reply({{"ok", true}, {"text", apply_template(tmpl, read_messages(req), req.value("add_assistant", true))}});
            } else if (op == "tokenize") {
                const std::string text = req.at("text").get<std::string>();
                reply({{"ok", true}, {"tokens", tokenize(vocab, text, req.value("add_special", true), req.value("parse_special", true))}});
            } else if (op == "count") {
                const std::string text = apply_template(tmpl, read_messages(req), true);
                reply({{"ok", true}, {"n_tokens", tokenize(vocab, text, true, true).size()}});
            } else if (op == "generate") {
                if (ctx == nullptr) throw std::runtime_error("generate is not available with --vocab-only");
                reply(generate(ctx, vocab, tmpl, req, opt.n_batch));
            } else {
                throw std::runtime_error("unknown op " + op);
            }
        } catch (const std::exception & e) {
            reply({{"ok", false}, {"error", e.what()}});
        }
    }

    if (ctx) llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
