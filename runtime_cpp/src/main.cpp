// anytime_runtime: a minimal C++ inference worker.
//
// Loads one or more ONNX models (one per "variant" name) and accepts
// line-delimited JSON requests on stdin. Each request specifies the variant
// and the input tensors; the worker runs ONNX Runtime and writes a response
// line containing the output tensor and the measured per-request latency.
//
// Protocol (one JSON object per line):
//   request : {"request_id": "...", "variant": "fp32",
//              "inputs": {"<name>": {"shape": [...], "dtype": "...",
//                                     "data": "<base64>"}}}
//   response: {"request_id": "...", "logits": {"shape": [...],
//              "dtype": "float32", "data": "<base64>"}, "latency_ms": 12.3}
//
// On startup, after every model is loaded, the worker prints the literal
// string "ready" followed by a newline. The Python client uses this as a
// handshake.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

std::string base64_encode(const uint8_t* data, std::size_t length) {
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((length + 2) / 3) * 4);
    std::size_t i = 0;
    for (; i + 3 <= length; i += 3) {
        const uint32_t value = (static_cast<uint32_t>(data[i]) << 16) |
                               (static_cast<uint32_t>(data[i + 1]) << 8) |
                                static_cast<uint32_t>(data[i + 2]);
        out.push_back(kAlphabet[(value >> 18) & 0x3F]);
        out.push_back(kAlphabet[(value >> 12) & 0x3F]);
        out.push_back(kAlphabet[(value >> 6) & 0x3F]);
        out.push_back(kAlphabet[value & 0x3F]);
    }
    if (i < length) {
        const std::size_t remaining = length - i;
        uint32_t value = static_cast<uint32_t>(data[i]) << 16;
        if (remaining == 2) {
            value |= static_cast<uint32_t>(data[i + 1]) << 8;
        }
        out.push_back(kAlphabet[(value >> 18) & 0x3F]);
        out.push_back(kAlphabet[(value >> 12) & 0x3F]);
        out.push_back(remaining == 2 ? kAlphabet[(value >> 6) & 0x3F] : '=');
        out.push_back('=');
    }
    return out;
}

std::vector<uint8_t> base64_decode(const std::string& input) {
    static int8_t table[256];
    static bool initialised = false;
    if (!initialised) {
        std::fill(std::begin(table), std::end(table), int8_t{-1});
        static const char* alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) {
            table[static_cast<unsigned char>(alphabet[i])] = static_cast<int8_t>(i);
        }
        initialised = true;
    }
    std::vector<uint8_t> out;
    out.reserve((input.size() / 4) * 3);
    uint32_t buffer = 0;
    int bits = 0;
    for (char c : input) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
        const int8_t value = table[static_cast<unsigned char>(c)];
        if (value < 0) throw std::runtime_error("invalid base64 character");
        buffer = (buffer << 6) | static_cast<uint32_t>(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buffer >> bits) & 0xFF));
        }
    }
    return out;
}

// Tiny JSON reader sufficient for the request envelope produced by the Python
// client. Strings are unescaped only for the characters the client emits
// (\\ and \"). Numbers are parsed as doubles. Arrays of numbers are returned
// as std::vector<int64_t>. Anything more exotic raises std::runtime_error.
class JsonReader {
public:
    explicit JsonReader(const std::string& source) : source_(source), pos_(0) {}

    struct Value {
        enum class Kind { Null, Bool, Number, String, Array, Object };
        Kind kind = Kind::Null;
        bool boolean = false;
        double number = 0.0;
        std::string string;
        std::vector<Value> array;
        std::vector<std::pair<std::string, Value>> object;

        const Value& find(const std::string& key) const {
            for (const auto& [k, v] : object) {
                if (k == key) return v;
            }
            throw std::runtime_error("missing JSON key: " + key);
        }
        std::vector<int64_t> as_int_array() const {
            std::vector<int64_t> out;
            out.reserve(array.size());
            for (const auto& v : array) {
                out.push_back(static_cast<int64_t>(v.number));
            }
            return out;
        }
    };

    Value parse() {
        skip_ws();
        Value v = parse_value();
        skip_ws();
        if (pos_ != source_.size()) {
            throw std::runtime_error("trailing characters in JSON");
        }
        return v;
    }

private:
    void skip_ws() {
        while (pos_ < source_.size() && std::isspace(static_cast<unsigned char>(source_[pos_]))) {
            ++pos_;
        }
    }

    Value parse_value() {
        skip_ws();
        if (pos_ >= source_.size()) throw std::runtime_error("unexpected end of JSON");
        char c = source_[pos_];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') { expect_literal("null"); Value v; v.kind = Value::Kind::Null; return v; }
        return parse_number();
    }

    Value parse_object() {
        Value v;
        v.kind = Value::Kind::Object;
        ++pos_;  // consume '{'
        skip_ws();
        if (pos_ < source_.size() && source_[pos_] == '}') { ++pos_; return v; }
        while (true) {
            skip_ws();
            Value key = parse_string();
            skip_ws();
            if (pos_ >= source_.size() || source_[pos_] != ':') {
                throw std::runtime_error("expected ':' in JSON object");
            }
            ++pos_;
            Value value = parse_value();
            v.object.emplace_back(std::move(key.string), std::move(value));
            skip_ws();
            if (pos_ < source_.size() && source_[pos_] == ',') { ++pos_; continue; }
            if (pos_ < source_.size() && source_[pos_] == '}') { ++pos_; break; }
            throw std::runtime_error("expected ',' or '}' in JSON object");
        }
        return v;
    }

    Value parse_array() {
        Value v;
        v.kind = Value::Kind::Array;
        ++pos_;  // consume '['
        skip_ws();
        if (pos_ < source_.size() && source_[pos_] == ']') { ++pos_; return v; }
        while (true) {
            v.array.push_back(parse_value());
            skip_ws();
            if (pos_ < source_.size() && source_[pos_] == ',') { ++pos_; continue; }
            if (pos_ < source_.size() && source_[pos_] == ']') { ++pos_; break; }
            throw std::runtime_error("expected ',' or ']' in JSON array");
        }
        return v;
    }

    Value parse_string() {
        Value v;
        v.kind = Value::Kind::String;
        if (source_[pos_] != '"') throw std::runtime_error("expected '\"' in JSON string");
        ++pos_;
        std::string out;
        while (pos_ < source_.size() && source_[pos_] != '"') {
            char c = source_[pos_++];
            if (c == '\\' && pos_ < source_.size()) {
                char escape = source_[pos_++];
                switch (escape) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    default: throw std::runtime_error("unsupported JSON escape");
                }
            } else {
                out.push_back(c);
            }
        }
        if (pos_ >= source_.size()) throw std::runtime_error("unterminated JSON string");
        ++pos_;  // closing quote
        v.string = std::move(out);
        return v;
    }

    Value parse_bool() {
        Value v;
        v.kind = Value::Kind::Bool;
        if (source_[pos_] == 't') {
            expect_literal("true");
            v.boolean = true;
        } else {
            expect_literal("false");
            v.boolean = false;
        }
        return v;
    }

    Value parse_number() {
        std::size_t start = pos_;
        if (source_[pos_] == '-') ++pos_;
        while (pos_ < source_.size() && (std::isdigit(static_cast<unsigned char>(source_[pos_])) ||
                                          source_[pos_] == '.' || source_[pos_] == 'e' ||
                                          source_[pos_] == 'E' || source_[pos_] == '+' ||
                                          source_[pos_] == '-')) {
            ++pos_;
        }
        Value v;
        v.kind = Value::Kind::Number;
        v.number = std::stod(source_.substr(start, pos_ - start));
        return v;
    }

    void expect_literal(const char* literal) {
        const std::size_t n = std::strlen(literal);
        if (pos_ + n > source_.size() || source_.compare(pos_, n, literal) != 0) {
            throw std::runtime_error(std::string("expected literal ") + literal);
        }
        pos_ += n;
    }

    const std::string& source_;
    std::size_t pos_;
};

std::string escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default: out.push_back(c);
        }
    }
    return out;
}

struct LoadedModel {
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

LoadedModel load_model(Ort::Env& env, const std::string& path) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    LoadedModel model;
    model.session = std::make_unique<Ort::Session>(env, path.c_str(), options);

    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t num_inputs = model.session->GetInputCount();
    for (std::size_t i = 0; i < num_inputs; ++i) {
        auto name = model.session->GetInputNameAllocated(i, allocator);
        model.input_names.emplace_back(name.get());
    }
    const std::size_t num_outputs = model.session->GetOutputCount();
    for (std::size_t i = 0; i < num_outputs; ++i) {
        auto name = model.session->GetOutputNameAllocated(i, allocator);
        model.output_names.emplace_back(name.get());
    }
    return model;
}

Ort::Value make_tensor(const JsonReader::Value& payload,
                       Ort::MemoryInfo& memory,
                       std::vector<uint8_t>& raw_storage,
                       std::vector<int64_t>& shape_storage,
                       std::vector<float>& float_storage,
                       std::vector<int64_t>& int64_storage) {
    const std::string dtype = payload.find("dtype").string;
    shape_storage = payload.find("shape").as_int_array();
    raw_storage = base64_decode(payload.find("data").string);

    if (dtype == "float32") {
        const std::size_t count = raw_storage.size() / sizeof(float);
        float_storage.resize(count);
        std::memcpy(float_storage.data(), raw_storage.data(), raw_storage.size());
        return Ort::Value::CreateTensor<float>(memory, float_storage.data(), count,
                                               shape_storage.data(), shape_storage.size());
    }
    if (dtype == "int64") {
        const std::size_t count = raw_storage.size() / sizeof(int64_t);
        int64_storage.resize(count);
        std::memcpy(int64_storage.data(), raw_storage.data(), raw_storage.size());
        return Ort::Value::CreateTensor<int64_t>(memory, int64_storage.data(), count,
                                                  shape_storage.data(), shape_storage.size());
    }
    throw std::runtime_error("unsupported dtype: " + dtype);
}

std::string serialise_output(const Ort::Value& tensor) {
    auto info = tensor.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = info.GetShape();
    const std::size_t element_count = info.GetElementCount();
    const auto element_type = info.GetElementType();
    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("only float32 output tensors are supported");
    }
    const float* data = tensor.GetTensorData<float>();
    const std::string encoded = base64_encode(
        reinterpret_cast<const uint8_t*>(data), element_count * sizeof(float));

    std::ostringstream shape_json;
    shape_json << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) shape_json << ',';
        shape_json << shape[i];
    }
    shape_json << ']';

    std::ostringstream out;
    out << "{\"shape\":" << shape_json.str()
        << ",\"dtype\":\"float32\""
        << ",\"data\":\"" << encoded << "\"}";
    return out.str();
}

struct ParsedArgs {
    std::vector<std::pair<std::string, std::string>> models;
};

ParsedArgs parse_args(int argc, char** argv) {
    ParsedArgs out;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--model" && i + 1 < argc) {
            std::string value = argv[++i];
            const std::size_t eq = value.find('=');
            if (eq == std::string::npos) {
                throw std::runtime_error("--model expects <variant>=<path>");
            }
            out.models.emplace_back(value.substr(0, eq), value.substr(eq + 1));
        } else {
            throw std::runtime_error("unknown argument: " + flag);
        }
    }
    if (out.models.empty()) {
        throw std::runtime_error("at least one --model must be specified");
    }
    return out;
}

}  // namespace


int main(int argc, char** argv) {
    try {
        const ParsedArgs args = parse_args(argc, argv);

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "anytime_runtime");
        std::unordered_map<std::string, LoadedModel> models;
        for (const auto& [variant, path] : args.models) {
            models[variant] = load_model(env, path);
        }

        std::cout << "ready" << std::endl;

        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;

            // Per-request failures must not take the worker down: a pool shares
            // one process per worker, so exiting here would fail every in-flight
            // and subsequent request on this worker. Anything recoverable is
            // reported as an error response and the loop continues.
            std::string request_id;
            try {
                JsonReader reader(line);
                JsonReader::Value request = reader.parse();

                request_id = request.find("request_id").string;
                const std::string variant = request.find("variant").string;
                const JsonReader::Value& inputs = request.find("inputs");

                const auto it = models.find(variant);
                if (it == models.end()) {
                    std::cerr << "unknown variant: " << variant << std::endl;
                    std::cout << "{\"request_id\":\"" << escape_json(request_id)
                              << "\",\"error\":\"unknown variant\"}" << std::endl;
                    continue;
                }
                LoadedModel& model = it->second;

                std::vector<std::vector<uint8_t>> raw_storage(inputs.object.size());
                std::vector<std::vector<int64_t>> shape_storage(inputs.object.size());
                std::vector<std::vector<float>> float_storage(inputs.object.size());
                std::vector<std::vector<int64_t>> int64_storage(inputs.object.size());
                std::vector<Ort::Value> tensors;
                tensors.reserve(inputs.object.size());
                std::vector<const char*> input_name_ptrs;
                input_name_ptrs.reserve(inputs.object.size());

                // Variants of the same task can declare different inputs: a
                // DistilBERT graph takes input_ids and attention_mask, a BERT
                // graph additionally takes token_type_ids. The caller sends the
                // union, so drop anything this graph does not declare rather than
                // failing the request.
                for (std::size_t i = 0; i < inputs.object.size(); ++i) {
                    const auto& [name, payload] = inputs.object[i];
                    const bool declared = std::find(model.input_names.begin(),
                                                    model.input_names.end(),
                                                    name) != model.input_names.end();
                    if (!declared) continue;
                    input_name_ptrs.push_back(name.c_str());
                    tensors.push_back(make_tensor(payload, memory,
                                                   raw_storage[i], shape_storage[i],
                                                   float_storage[i], int64_storage[i]));
                }
                if (tensors.size() != model.input_names.size()) {
                    throw std::runtime_error(
                        "request supplies " + std::to_string(tensors.size()) +
                        " of the " + std::to_string(model.input_names.size()) +
                        " inputs this graph declares");
                }

                std::vector<const char*> output_name_ptrs;
                output_name_ptrs.reserve(model.output_names.size());
                for (const auto& n : model.output_names) output_name_ptrs.push_back(n.c_str());

                const auto start = std::chrono::steady_clock::now();
                auto outputs = model.session->Run(
                    Ort::RunOptions{nullptr},
                    input_name_ptrs.data(), tensors.data(), tensors.size(),
                    output_name_ptrs.data(), output_name_ptrs.size());
                const auto end = std::chrono::steady_clock::now();
                const double latency_ms =
                    std::chrono::duration<double, std::milli>(end - start).count();

                std::string logits_json = serialise_output(outputs.front());
                std::cout << "{\"request_id\":\"" << escape_json(request_id)
                          << "\",\"logits\":" << logits_json
                          << ",\"latency_ms\":" << latency_ms << "}" << std::endl;
            } catch (const std::exception& exc) {
                std::cerr << "request error: " << exc.what() << std::endl;
                std::cout << "{\"request_id\":\"" << escape_json(request_id)
                          << "\",\"error\":\"" << escape_json(exc.what()) << "\"}"
                          << std::endl;
            }
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "fatal: " << exc.what() << std::endl;
        return 1;
    }
}
