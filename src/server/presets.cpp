#include "server/presets.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qv::server {

namespace {

std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && (unsigned char)s[a] <= ' ') ++a;
    size_t b = s.size();
    while (b > a && (unsigned char)s[b - 1] <= ' ') --b;
    return s.substr(a, b - a);
}

std::string unquote(const std::string& s) {
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\'')))
        return s.substr(1, s.size() - 2);
    return s;
}

std::string norm_key(std::string s) {
    for (char& ch : s) {
        if (ch == '_') ch = '-';
        else ch = (char)std::tolower((unsigned char)ch);
    }
    return s;
}

bool is_true(const std::string& v) {
    std::string l = norm_key(v);
    return l == "1" || l == "true" || l == "on" || l == "yes" || l == "enabled";
}

bool is_false(const std::string& v) {
    std::string l = norm_key(v);
    return l == "0" || l == "false" || l == "off" || l == "no" || l == "disabled";
}

using KVList = std::vector<std::pair<std::string, std::string>>;

void apply_kv(Options& o, const std::string& sec, const std::string& key, const std::string& val, std::vector<std::string>& warnings, std::set<std::string>& warned) {
    auto warn_once = [&](const std::string& tag, const std::string& m) {
        if (warned.insert(sec + ":" + tag).second) warnings.push_back("preset [" + sec + "]: " + m);
    };
    std::string msg;
    if (key == "max-context" || key == "kv-capacity") {
        if (!apply_preset_option(o, "ctx-size", val, msg) && !msg.empty()) warn_once(key, msg);
        return;
    }
    if (key == "kv-dtype") {
        std::string v = norm_key(val);
        if (v == "fp32" || v == "fp16" || v == "int4") {
            apply_preset_option(o, "kv", v, msg);
        } else {
            warn_once(key, "kv-dtype '" + val + "' is not supported by qv_server (supported: fp32, fp16, int4); falling back to int4");
            apply_preset_option(o, "kv", "int4", msg);
        }
        return;
    }
    if (key == "spec") {
        std::string v = norm_key(val);
        if (v == "mtp") apply_preset_option(o, "mtp", "true", msg);
        else if (is_false(v) || v == "none") apply_preset_option(o, "mtp", "false", msg);
        else warn_once(key, "speculative mode '" + val + "' is not supported by qv_server (only 'mtp' is supported; ignored)");
        return;
    }
    if (key == "draft-tokens") {
        if (!apply_preset_option(o, "mtp-depth", val, msg) && !msg.empty()) warn_once(key, msg);
        return;
    }
    if (key == "default-reasoning-effort") {
        if (!apply_preset_option(o, "reasoning-effort", val, msg) && !msg.empty()) warn_once(key, msg);
        return;
    }
    if (key == "default-max-tokens") {
        if (!apply_preset_option(o, "predict", val, msg) && !msg.empty()) warn_once(key, msg);
        return;
    }
    if (key == "default-thinking-budget") {
        if (!apply_preset_option(o, "reasoning-budget", val, msg) && !msg.empty()) warn_once(key, msg);
        return;
    }
    if (key == "max-pending-requests") {
        if (!apply_preset_option(o, "max-queue", val, msg) && !msg.empty()) warn_once(key, msg);
        return;
    }
    if (key == "max-concurrency") {
        if (!apply_preset_option(o, "parallel", val, msg) && !msg.empty()) warn_once(key, msg);
        return;
    }
    if (key == "vision") {
        if (is_false(val)) o.vision_file = "none";
        else if (is_true(val)) { if (o.vision_file == "none") o.vision_file.clear(); }
        else warn_once(key, "invalid boolean '" + val + "' for vision (ignored)");
        return;
    }
    if (key == "preload" || key == "load-on-startup") {
        if (is_true(val)) o.preload = true;
        else if (is_false(val)) o.preload = false;
        else warn_once(key, "invalid boolean '" + val + "' for " + key + " (ignored)");
        return;
    }
    if (!apply_preset_option(o, key, val, msg)) {
        warn_once(key, msg.empty() ? ("option '" + key + "' is not supported by qv_server (ignored)") : msg);
    } else if (!msg.empty()) {
        warn_once(key, msg);
    }
}

}  // namespace

std::vector<ModelPreset> load_presets_ini(const std::string& path, const Options& base, std::vector<std::string>& warnings) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("failed to open presets file '" + path + "'");

    KVList global_kvs;
    std::vector<std::string> order;
    std::map<std::string, KVList> sections;
    std::string cur;
    std::string line;
    int lno = 0;

    while (std::getline(f, line)) {
        ++lno;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::string s = trim(line);
        if (s.empty() || s[0] == ';' || s[0] == '#') continue;
        if (s.front() == '[') {
            if (s.back() != ']') throw std::runtime_error(path + ":" + std::to_string(lno) + ": unclosed section header");
            cur = trim(s.substr(1, s.size() - 2));
            if (cur.empty()) throw std::runtime_error(path + ":" + std::to_string(lno) + ": empty section name");
            if (cur != "*" && !sections.count(cur)) {
                order.push_back(cur);
                sections[cur] = {};
            }
            continue;
        }
        size_t eq = s.find('=');
        if (eq == std::string::npos) throw std::runtime_error(path + ":" + std::to_string(lno) + ": expected 'key = value'");
        std::string k = norm_key(trim(s.substr(0, eq)));
        std::string v = unquote(trim(s.substr(eq + 1)));
        if (k.empty()) throw std::runtime_error(path + ":" + std::to_string(lno) + ": empty key");
        if (cur.empty() || cur == "*") global_kvs.emplace_back(std::move(k), std::move(v));
        else sections[cur].emplace_back(std::move(k), std::move(v));
    }

    if (order.empty()) throw std::runtime_error("presets file '" + path + "' defines no model sections");

    std::vector<ModelPreset> out;
    std::set<std::string> warned;
    for (const std::string& id : order) {
        Options o = base;
        o.model.clear();
        o.tokenizer.clear();
        o.vision_file.clear();
        o.mtp_file.clear();
        o.alias.clear();
        o.warnings.clear();
        o.preload = false;
        o.given.erase("model");
        o.given.erase("tokenizer");
        o.given.erase("alias");
        for (const auto& [k, v] : global_kvs) apply_kv(o, "*", k, v, warnings, warned);
        for (const auto& [k, v] : sections[id]) apply_kv(o, id, k, v, warnings, warned);
        if (o.model.empty()) throw std::runtime_error("preset [" + id + "] has no 'model' path");
        out.push_back(ModelPreset{id, std::move(o)});
    }
    return out;
}

}  // namespace qv::server
