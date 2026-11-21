#include "grammar/json_schema.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace qv::grammar {
namespace {

const std::string SPACE_RULE = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";
constexpr int RECURSIVE_LIMIT = 1024;

struct BuiltinRule {
    std::string content;
    std::vector<std::string> deps;
};
const std::unordered_map<std::string, BuiltinRule>& primitive_rules() {
    static const std::unordered_map<std::string, BuiltinRule> m = {
        {"boolean", {"(\"true\" | \"false\")", {}}},
        {"decimal-part", {"[0-9]{1,16}", {}}},
        {"integral-part", {"[0] | [1-9] [0-9]{0,15}", {}}},
        {"number", {"(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?", {"integral-part", "decimal-part"}}},
        {"integer", {"(\"-\"? integral-part)", {"integral-part"}}},
        {"value", {"object | array | string | number | boolean | null", {"object", "array", "string", "number", "boolean", "null"}}},
        {"object", {"\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"", {"string", "value"}}},
        {"array", {"\"[\" space ( value (\",\" space value)* )? space \"]\"", {"value"}}},
        {"uuid", {"\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\"", {}}},
        {"char", {"[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", {}}},
        {"string", {"\"\\\"\" char* \"\\\"\"", {"char"}}},
        {"null", {"\"null\"", {}}},
    };
    return m;
}
const std::unordered_map<std::string, BuiltinRule>& format_rules() {
    static const std::unordered_map<std::string, BuiltinRule> m = {
        {"date", {"[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", {}}},
        {"time", {"([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", {}}},
        {"date-time", {"date \"T\" time", {"date", "time"}}},
        {"date-string", {"\"\\\"\" date \"\\\"\"", {"date"}}},
        {"time-string", {"\"\\\"\" time \"\\\"\"", {"time"}}},
        {"date-time-string", {"\"\\\"\" date-time \"\\\"\"", {"date-time"}}},
    };
    return m;
}
bool is_reserved_name(const std::string& n) {
    return n == "root" || primitive_rules().count(n) || format_rules().count(n);
}

std::string repeat_str(const std::string& s, size_t n) {
    std::string o;
    for (size_t i = 0; i < n; ++i) o += s;
    return o;
}

std::string build_repetition(const std::string& item, int min_items, int max_items, const std::string& sep = "") {
    bool has_max = max_items != std::numeric_limits<int>::max();
    if (max_items == 0) return "";
    if (min_items == 0 && max_items == 1) return item + "?";
    if (sep.empty()) {
        if (min_items == 1 && !has_max) return item + "+";
        if (min_items == 0 && !has_max) return item + "*";
        return item + "{" + std::to_string(min_items) + "," + (has_max ? std::to_string(max_items) : "") + "}";
    }
    auto result = item + " " + build_repetition("(" + sep + " " + item + ")", min_items == 0 ? 0 : min_items - 1, has_max ? max_items - 1 : max_items);
    if (min_items == 0) result = "(" + result + ")?";
    return result;
}

// ---- integer ranges (port of llama.cpp build_min_max_int)
void build_min_max_int(int64_t min_value, int64_t max_value, std::stringstream& out, int decimals_left = 16, bool top_level = true) {
    constexpr int64_t I64MIN = std::numeric_limits<int64_t>::min(), I64MAX = std::numeric_limits<int64_t>::max();
    bool has_min = min_value != I64MIN, has_max = max_value != I64MAX;
    auto digit_range = [&](char from, char to) {
        out << "[";
        if (from == to) out << from;
        else out << from << "-" << to;
        out << "]";
    };
    auto more_digits = [&](int min_digits, int max_digits) {
        out << "[0-9]";
        if (min_digits == max_digits && min_digits == 1) return;
        out << "{" << min_digits;
        if (max_digits != min_digits) {
            out << ",";
            if (max_digits != std::numeric_limits<int>::max()) out << max_digits;
        }
        out << "}";
    };
    std::function<void(const std::string_view&, const std::string_view&)> uniform_range = [&](const std::string_view& from, const std::string_view& to) {
        size_t i = 0;
        while (i < from.length() && i < to.length() && from[i] == to[i]) i++;
        if (i > 0) out << "\"" << from.substr(0, i) << "\"";
        if (i < from.length() && i < to.length()) {
            if (i > 0) out << " ";
            auto sub_len = from.length() - i - 1;
            if (sub_len > 0) {
                auto from_sub = from.substr(i + 1), to_sub = to.substr(i + 1);
                auto sub_zeros = repeat_str("0", sub_len), sub_nines = repeat_str("9", sub_len);
                bool to_reached = false;
                out << "(";
                if (from_sub == sub_zeros) {
                    digit_range(from[i], to[i] - 1);
                    out << " ";
                    more_digits((int)sub_len, (int)sub_len);
                } else {
                    out << "[" << from[i] << "] (";
                    uniform_range(from_sub, sub_nines);
                    out << ")";
                    if (from[i] < to[i] - 1) {
                        out << " | ";
                        if (to_sub == sub_nines) { digit_range(from[i] + 1, to[i]); to_reached = true; }
                        else digit_range(from[i] + 1, to[i] - 1);
                        out << " ";
                        more_digits((int)sub_len, (int)sub_len);
                    }
                }
                if (!to_reached) {
                    out << " | ";
                    digit_range(to[i], to[i]);
                    out << " ";
                    uniform_range(sub_zeros, to_sub);
                }
                out << ")";
            } else out << "[" << from[i] << "-" << to[i] << "]";
        }
    };
    if (has_min && has_max) {
        if (min_value < 0 && max_value < 0) {
            out << "\"-\" (";
            build_min_max_int(-max_value, -min_value, out, decimals_left, true);
            out << ")";
            return;
        }
        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(0, -min_value, out, decimals_left, true);
            out << ") | ";
            min_value = 0;
        }
        auto min_s = std::to_string(min_value), max_s = std::to_string(max_value);
        for (auto digits = min_s.length(); digits < max_s.length(); digits++) {
            uniform_range(min_s, repeat_str("9", digits));
            min_s = "1" + repeat_str("0", digits);
            out << " | ";
        }
        uniform_range(min_s, max_s);
        return;
    }
    auto less_decimals = std::max(decimals_left - 1, 1);
    if (has_min) {
        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(I64MIN, -min_value, out, decimals_left, false);
            out << ") | [0] | [1-9] ";
            more_digits(0, decimals_left - 1);
        } else if (min_value == 0) {
            if (top_level) { out << "[0] | [1-9] "; more_digits(0, less_decimals); }
            else more_digits(1, decimals_left);
        } else if (min_value <= 9) {
            char c = (char)('0' + min_value);
            char range_start = top_level ? '1' : '0';
            if (c > range_start) {
                digit_range(range_start, c - 1);
                out << " ";
                more_digits(1, less_decimals);
                out << " | ";
            }
            digit_range(c, '9');
            out << " ";
            more_digits(0, less_decimals);
        } else {
            auto min_s = std::to_string(min_value);
            auto len = min_s.length();
            char c = min_s[0];
            if (c > '1') {
                digit_range(top_level ? '1' : '0', c - 1);
                out << " ";
                more_digits((int)len, less_decimals);
                out << " | ";
            }
            digit_range(c, c);
            out << " (";
            build_min_max_int(std::stoll(min_s.substr(1)), I64MAX, out, less_decimals, false);
            out << ")";
            if (c < '9') {
                out << " | ";
                digit_range(c + 1, '9');
                out << " ";
                more_digits((int)len - 1, less_decimals);
            }
        }
        return;
    }
    if (has_max) {
        if (max_value >= 0) {
            if (top_level) {
                out << "\"-\" [1-9] ";
                more_digits(0, less_decimals);
                out << " | ";
            }
            build_min_max_int(0, max_value, out, decimals_left, true);
        } else {
            out << "\"-\" (";
            build_min_max_int(-max_value, I64MAX, out, decimals_left, false);
            out << ")";
        }
        return;
    }
    throw std::runtime_error("At least one of min_value or max_value must be set");
}

size_t gbnf_escape_length(const std::string& pattern, size_t pos) {
    if (pos + 1 >= pattern.length() || pattern[pos] != '\\') return 0;
    size_t n_hex = 0;
    switch (pattern[pos + 1]) {
        case 'x': n_hex = 2; break;
        case 'u': n_hex = 4; break;
        case 'U': n_hex = 8; break;
        case 't': case 'r': case 'n': case '\\': case '"': case '[': case ']': case '-': return 2;
        default: return 0;
    }
    if (pos + 2 + n_hex > pattern.length()) return 0;
    for (size_t i = pos + 2; i < pos + 2 + n_hex; i++)
        if (!isxdigit((unsigned char)pattern[i])) return 0;
    return 2 + n_hex;
}

const std::unordered_set<char> NON_LITERAL_SET = {'|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?', '^', '$'};
const std::unordered_set<char> ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = {'^', '$', '.', '[', ']', '(', ')', '|', '{', '}', '*', '+', '?'};

std::vector<std::string> split_str(const std::string& s, char d) {
    std::vector<std::string> o;
    size_t p = 0;
    for (;;) {
        size_t f = s.find(d, p);
        if (f == std::string::npos) { o.push_back(s.substr(p)); break; }
        o.push_back(s.substr(p, f - p));
        p = f + 1;
    }
    return o;
}
std::string join_str(const std::vector<std::string>& v, const std::string& sep) {
    std::string o;
    for (size_t i = 0; i < v.size(); ++i) { if (i) o += sep; o += v[i]; }
    return o;
}
void append_utf8(std::string& s, uint32_t c) {
    if (c < 0x80) s += (char)c;
    else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 63)); }
    else if (c < 0x10000) { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 63)); s += (char)(0x80 | (c & 63)); }
    else { s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 63)); s += (char)(0x80 | ((c >> 6) & 63)); s += (char)(0x80 | (c & 63)); }
}
std::vector<uint32_t> decode_cps(const std::string& s) {
    std::vector<uint32_t> o;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
        uint32_t cp = n == 1 ? c : c & (0xFF >> (n + 1));
        for (int k = 1; k < n && i + k < s.size(); ++k) cp = (cp << 6) | (s[i + k] & 63);
        o.push_back(cp);
        i += n;
    }
    return o;
}

struct unsupported_pattern : std::runtime_error { using std::runtime_error::runtime_error; };
struct invalid_pattern : std::runtime_error { using std::runtime_error::runtime_error; };

}  // namespace

std::string gbnf_literal(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        switch (c) {
            case '\r': o += "\\r"; break;
            case '\n': o += "\\n"; break;
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            default: o += c;
        }
    }
    return o + "\"";
}
std::string gbnf_rule_name(const std::string& name) {
    std::string out;
    bool in_invalid = false;
    for (char c : name) {
        if (isalnum((unsigned char)c) || c == '-') { out += c; in_invalid = false; }
        else if (!in_invalid) { out += '-'; in_invalid = true; }
    }
    return out;
}

// ============================================================================ converter
struct SchemaGrammarBuilder::Conv {
    SchemaGrammarBuilder& b;
    jd::Value root;
    std::unordered_set<std::string> refs_being_resolved;
    std::map<std::string, std::string> ref_rules;
    std::vector<std::string> errors;
    int depth = 0;

    struct Guard {
        int& d;
        explicit Guard(int& d_) : d(d_) {
            if (++d > RECURSIVE_LIMIT) throw std::invalid_argument("JSON schema is nested more than " + std::to_string(RECURSIVE_LIMIT) + " levels deep");
        }
        ~Guard() { --d; }
    };

    Conv(SchemaGrammarBuilder& bb, const jd::Value& r) : b(bb), root(r) {}

    std::string add_rule(const std::string& name, const std::string& rule) { return b.add_rule(name, rule); }

    std::string add_primitive(const std::string& name, const BuiltinRule& rule) {
        auto n = add_rule(name, rule.content);
        for (const auto& dep : rule.deps) {
            auto it = primitive_rules().find(dep);
            if (it == primitive_rules().end()) {
                auto it2 = format_rules().find(dep);
                if (it2 == format_rules().end()) { errors.push_back("Rule " + dep + " not known"); continue; }
                if (!b.rules_.count(dep)) add_primitive(dep, it2->second);
            } else if (!b.rules_.count(dep)) add_primitive(dep, it->second);
        }
        return n;
    }
    std::string prim(const std::string& name) { return add_primitive(name, primitive_rules().at(name)); }
    std::string visit_primitive(const std::string& rule_name, const std::string& type) {
        return add_primitive(rule_name == "root" ? "root" : type, primitive_rules().at(type));
    }

    std::string generate_union_rule(const std::string& name, const std::vector<const jd::Value*>& alts) {
        std::vector<std::string> rules;
        for (size_t i = 0; i < alts.size(); ++i) rules.push_back(visit(*alts[i], name + (name.empty() ? "alternative-" : "-") + std::to_string(i)));
        return join_str(rules, " | ");
    }

    // ---- regex patterns
    std::string visit_pattern(const std::string& pattern, const std::string& name) {
        auto snapshot = b.rules_;
        try {
            return pattern_to_rule(pattern, name);
        } catch (const unsupported_pattern& e) {
            b.rules_ = std::move(snapshot);
            b.warnings_.push_back("pattern " + pattern + " is not supported (" + e.what() + "), accepting any string");
            return add_rule(name, prim("string"));
        } catch (const invalid_pattern& e) {
            b.rules_ = std::move(snapshot);
            errors.push_back("Invalid pattern " + pattern + ": " + e.what());
            return "";
        }
    }
    std::string pattern_to_rule(const std::string& pattern, const std::string& name) {
        if (pattern.length() < 2 || pattern.front() != '^' || pattern.back() != '$') throw unsupported_pattern("not anchored with '^' and '$'");
        std::string sub_pattern = pattern.substr(1, pattern.length() - 2);
        std::unordered_map<std::string, std::string> sub_rule_ids;
        size_t i = 0, length = sub_pattern.length();
        int paren_depth = 0;
        using lor = std::pair<std::string, bool>;
        auto to_rule = [&](const lor& ls) { return ls.second ? "\"" + ls.first + "\"" : ls.first; };
        std::function<lor()> transform = [&]() -> lor {
            std::vector<lor> seq;
            auto get_dot = [&]() { return add_rule("dot", b.opt_.dotall ? "[\\U00000000-\\U0010FFFF]" : "[^\\x0A\\x0D]"); };
            auto join_seq = [&]() {
                std::vector<lor> ret;
                std::string literal;
                auto flush = [&]() { if (literal.empty()) return; ret.emplace_back(literal, true); literal.clear(); };
                for (auto& item : seq) {
                    if (item.second) literal += item.first;
                    else { flush(); ret.push_back(item); }
                }
                flush();
                std::vector<std::string> results;
                for (auto& item : ret) results.push_back(to_rule(item));
                return lor(join_str(results, " "), false);
            };
            while (i < length) {
                char c = sub_pattern[i];
                if (c == '.') { seq.emplace_back(get_dot(), false); i++; }
                else if (c == '(') {
                    i++;
                    if (i < length && sub_pattern[i] == '?') {
                        if (i + 1 < length && sub_pattern[i + 1] == ':') i += 2;
                        else throw unsupported_pattern("unsupported group syntax");
                    }
                    paren_depth++;
                    if (paren_depth > RECURSIVE_LIMIT) throw unsupported_pattern("pattern nesting too deep");
                    seq.emplace_back("(" + to_rule(transform()) + ")", false);
                } else if (c == ')') {
                    i++;
                    if (paren_depth == 0) throw invalid_pattern("unbalanced parentheses");
                    paren_depth--;
                    return join_seq();
                } else if (c == '^' || c == '$') throw unsupported_pattern("anchor inside the pattern");
                else if (c == '[') {
                    std::string sb(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != ']') {
                        if (sub_pattern[i] == '\\') {
                            auto el = gbnf_escape_length(sub_pattern, i);
                            if (el == 0) throw unsupported_pattern("unsupported escape in character class: " + sub_pattern.substr(i, 2));
                            sb += sub_pattern.substr(i, el);
                            i += el;
                        } else { sb += sub_pattern[i]; i++; }
                    }
                    if (i >= length) throw invalid_pattern("unterminated character class");
                    sb += ']';
                    i++;
                    seq.emplace_back(sb, false);
                } else if (c == '|') { seq.emplace_back("|", false); i++; }
                else if (c == '*' || c == '+' || c == '?') {
                    if (seq.empty()) throw invalid_pattern("nothing to repeat");
                    seq.back() = std::make_pair(to_rule(seq.back()) + c, false);
                    i++;
                } else if (c == '{') {
                    std::string cb(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != '}') { cb += sub_pattern[i]; i++; }
                    if (i >= length) throw unsupported_pattern("unterminated curly brackets");
                    cb += '}';
                    i++;
                    auto nums = split_str(cb.substr(1, cb.length() - 2), ',');
                    int min_times = 0, max_times = std::numeric_limits<int>::max();
                    if (nums.size() != 1 && nums.size() != 2) throw unsupported_pattern("wrong number of values in curly brackets");
                    try {
                        if (nums.size() == 1) min_times = max_times = std::stoi(nums[0]);
                        else {
                            if (!nums[0].empty()) min_times = std::stoi(nums[0]);
                            if (!nums[1].empty()) max_times = std::stoi(nums[1]);
                        }
                    } catch (const std::logic_error&) { throw unsupported_pattern("invalid number in curly brackets"); }
                    if (seq.empty()) throw invalid_pattern("nothing to repeat");
                    auto& last = seq.back();
                    auto& sub = last.first;
                    bool sub_is_literal = last.second;
                    if (!sub_is_literal) {
                        std::string& sub_id = sub_rule_ids[sub];
                        if (sub_id.empty()) sub_id = add_rule(name + "-" + std::to_string(sub_rule_ids.size()), sub);
                        sub = sub_id;
                    }
                    seq.back().first = build_repetition(sub_is_literal ? "\"" + sub + "\"" : sub, min_times, max_times, "");
                    seq.back().second = false;
                } else {
                    std::string literal;
                    auto non_lit = [&](char ch) { return NON_LITERAL_SET.count(ch) != 0; };
                    while (i < length) {
                        if (sub_pattern[i] == '\\') {
                            if (i == length - 1) throw invalid_pattern("trailing backslash");
                            char next = sub_pattern[i + 1];
                            if (ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.count(next)) { i++; literal += sub_pattern[i]; i++; }
                            else {
                                auto el = gbnf_escape_length(sub_pattern, i);
                                if (el == 0) throw unsupported_pattern("unsupported escape: " + sub_pattern.substr(i, 2));
                                literal += sub_pattern.substr(i, el);
                                i += el;
                            }
                        } else if (sub_pattern[i] == '"') { literal += "\\\""; i++; }
                        else if (!non_lit(sub_pattern[i]) && (i == length - 1 || literal.empty() || sub_pattern[i + 1] == '.' || !non_lit(sub_pattern[i + 1]))) { literal += sub_pattern[i]; i++; }
                        else break;
                    }
                    if (literal.empty()) throw unsupported_pattern(std::string("unsupported character: ") + c);
                    seq.emplace_back(literal, true);
                }
            }
            return join_seq();
        };
        auto rule = to_rule(transform());
        if (paren_depth != 0) throw invalid_pattern("unbalanced parentheses");
        return add_rule(name, "\"\\\"\" (" + rule + ") \"\\\"\"");
    }

    // a JSON string that is none of `strings` (for additionalProperties next to explicit properties)
    std::string not_strings(const std::vector<std::string>& strings) {
        struct TNode { std::map<uint32_t, int> children; bool terminal = false; };
        std::vector<TNode> trie(1);
        for (auto& s : strings) {
            int cur = 0;
            for (uint32_t cp : decode_cps(s)) {
                auto it = trie[cur].children.find(cp);
                if (it == trie[cur].children.end()) { trie.emplace_back(); int nid = (int)trie.size() - 1; trie[cur].children[cp] = nid; cur = nid; }
                else cur = it->second;
            }
            trie[cur].terminal = true;
        }
        std::string char_rule = prim("char");
        std::ostringstream out;
        out << "[\"] ( ";
        std::function<void(int, int)> visit_n = [&](int idx, int d) {
            if (d >= RECURSIVE_LIMIT) throw std::invalid_argument("JSON schema property name is too long");
            const auto& node = trie[idx];
            std::string rejects;
            bool first = true;
            for (auto& [cp, child] : node.children) {
                std::string c;
                append_utf8(c, cp);
                rejects += c;
                if (!first) out << " | ";
                first = false;
                out << "[" << c << "]";
                if (!trie[child].children.empty()) { out << " ("; visit_n(child, d + 1); out << ")"; }
                else out << " " << char_rule << "+";
            }
            if (!node.children.empty()) out << " | [^\"" << rejects << "] " << char_rule << "*";
        };
        visit_n(0, 0);
        out << " )";
        if (!trie[0].terminal) out << "?";
        out << " [\"]";
        return out.str();
    }

    // ---- $ref
    const jd::Value* resolve_pointer(const std::string& ref, std::string& fragment) {
        size_t h = ref.find('#');
        if (h == std::string::npos || h != 0) return nullptr;  // only local refs
        fragment = ref.substr(1);
        const jd::Value* cur = &root;
        std::string path = fragment;
        if (path.empty()) return cur;
        if (path[0] != '/') return nullptr;
        size_t p = 1;
        while (p <= path.size()) {
            size_t n = path.find('/', p);
            std::string tok = path.substr(p, n == std::string::npos ? std::string::npos : n - p);
            std::string un;
            for (size_t k = 0; k < tok.size(); ++k) {
                if (tok[k] == '~' && k + 1 < tok.size() && tok[k + 1] == '1') { un += '/'; ++k; }
                else if (tok[k] == '~' && k + 1 < tok.size() && tok[k + 1] == '0') { un += '~'; ++k; }
                else if (tok[k] == '%' && k + 2 < tok.size()) { un += (char)std::stoi(tok.substr(k + 1, 2), nullptr, 16); k += 2; }
                else un += tok[k];
            }
            if (cur->is_object()) { cur = cur->find(un); }
            else if (cur->is_array()) {
                size_t idx = (size_t)atoll(un.c_str());
                cur = idx < cur->items().size() ? &cur->items()[idx] : nullptr;
            } else cur = nullptr;
            if (!cur) return nullptr;
            if (n == std::string::npos) break;
            p = n + 1;
        }
        return cur;
    }
    std::string resolve_ref(const std::string& ref) {
        std::string fragment;
        std::string ref_name = "ref" + gbnf_rule_name(ref.find('#') != std::string::npos ? ref.substr(ref.find('#') + 1) : ref);
        if (!b.rules_.count(ref_name) && !refs_being_resolved.count(ref)) {
            const jd::Value* target = resolve_pointer(ref, fragment);
            if (!target) { errors.push_back("Unresolved $ref " + ref); return ""; }
            refs_being_resolved.insert(ref);
            jd::Value copy = *target;  // visit() may not alias into `root`
            ref_name = visit(copy, ref_name);
            refs_being_resolved.erase(ref);
        }
        return ref_name;
    }

    // ---- objects
    std::string build_object_rule(const std::vector<std::pair<std::string, jd::Value>>& properties, const std::set<std::string>& required, const std::string& name,
                                  const jd::Value* additional) {
        std::vector<std::string> required_props, optional_props, prop_names;
        std::map<std::string, std::string> kv_rule;
        for (auto& kv : properties) {
            const std::string& pn = kv.first;
            std::string prop_rule = visit(kv.second, name + (name.empty() ? "" : "-") + pn);
            kv_rule[pn] = add_rule(name + (name.empty() ? "" : "-") + pn + "-kv", gbnf_literal(jd::Value(pn).dump()) + " space \":\" space " + prop_rule);
            (required.count(pn) ? required_props : optional_props).push_back(pn);
            prop_names.push_back(pn);
        }
        if (additional) {
            std::string sub_name = name + (name.empty() ? "" : "-") + "additional";
            std::string value_rule;
            if (additional->is_object() && !additional->members().empty()) value_rule = visit(*additional, sub_name + "-value");
            else value_rule = prim("value");
            std::string key_rule = prop_names.empty() ? prim("string") : add_rule(sub_name + "-k", not_strings(prop_names));
            kv_rule["*"] = add_rule(sub_name + "-kv", key_rule + " \":\" space " + value_rule);
            optional_props.push_back("*");
        }
        if (required_props.empty() && optional_props.empty()) return "\"{\" space \"}\"";
        std::string rule = "\"{\" space ";
        for (size_t i = 0; i < required_props.size(); ++i) {
            if (i) rule += " \",\" space ";
            rule += kv_rule[required_props[i]];
        }
        if (!optional_props.empty()) {
            rule += " (";
            if (!required_props.empty()) rule += " \",\" space ( ";
            std::function<std::string(const std::vector<std::string>&, bool)> rec = [&](const std::vector<std::string>& ks, bool first_is_optional) {
                std::string res;
                if (ks.empty()) return res;
                const std::string& k = ks[0];
                std::string kv_name = kv_rule[k];
                std::string comma_ref = "( \",\" space " + kv_name + " )";
                if (first_is_optional) res = comma_ref + (k == "*" ? "*" : "?");
                else res = kv_name + (k == "*" ? " " + comma_ref + "*" : "");
                if (ks.size() > 1)
                    res += " " + add_rule(name + (name.empty() ? "" : "-") + k + "-rest", rec(std::vector<std::string>(ks.begin() + 1, ks.end()), true));
                return res;
            };
            for (size_t i = 0; i < optional_props.size(); ++i) {
                if (i) rule += " | ";
                rule += rec(std::vector<std::string>(optional_props.begin() + i, optional_props.end()), false);
            }
            if (!required_props.empty()) rule += " )";
            rule += " )?";
        }
        rule += " space \"}\"";
        return rule;
    }

    static const jd::Value* get(const jd::Value& o, const char* k) { return o.is_object() ? o.find(k) : nullptr; }
    static bool has(const jd::Value& o, const char* k) { return get(o, k) != nullptr; }

    std::string visit(const jd::Value& schema, const std::string& name) {
        Guard guard(depth);
        std::string rule_name = is_reserved_name(name) ? name + "-" : name.empty() ? "root" : name;
        std::string sub_name = name + (name.empty() ? "" : "-");
        if (schema.is_bool() || (schema.is_object() && schema.members().empty())) {
            if (schema.is_bool() && !schema.as_bool()) throw std::invalid_argument("schema `false` matches nothing");
            return add_rule(rule_name, prim("value"));
        }
        if (!schema.is_object()) throw std::invalid_argument("schema must be an object or boolean");
        const jd::Value* ty = get(schema, "type");
        std::string type = ty && ty->is_string() ? ty->as_string() : "";
        if (const jd::Value* ref = get(schema, "$ref"); ref && ref->is_string()) {
            return add_rule(rule_name, resolve_ref(ref->as_string()));
        }
        for (const char* key : {"oneOf", "anyOf"}) {
            const jd::Value* alts = get(schema, key);
            if (alts && alts->is_array()) {
                std::vector<const jd::Value*> v;
                for (auto& a : alts->items()) v.push_back(&a);
                return add_rule(rule_name, generate_union_rule(name, v));
            }
        }
        if (ty && ty->is_array()) {
            std::vector<jd::Value> copies;
            for (auto& t : ty->items()) {
                jd::Value c = jd::Value::object();
                for (auto& m : schema.members()) if (m.first != "type") c.set(m.first, m.second);
                c.set("type", t);
                copies.push_back(std::move(c));
            }
            std::vector<const jd::Value*> v;
            for (auto& c : copies) v.push_back(&c);
            return add_rule(rule_name, generate_union_rule(name, v));
        }
        if (const jd::Value* c = get(schema, "const")) return add_rule(rule_name, gbnf_literal(c->dump()));
        if (const jd::Value* en = get(schema, "enum"); en && en->is_array()) {
            std::vector<std::string> vals;
            for (auto& v : en->items()) vals.push_back(gbnf_literal(v.dump()));
            return add_rule(rule_name, "(" + join_str(vals, " | ") + ")");
        }
        const jd::Value* props = get(schema, "properties");
        const jd::Value* addl = get(schema, "additionalProperties");
        bool addl_true = addl && addl->is_bool() && addl->as_bool();
        if ((type.empty() || type == "object") && (props || (addl && !addl_true && !(addl->is_bool() && !addl->as_bool() && !props)))) {
            std::vector<std::pair<std::string, jd::Value>> pv;
            if (props && props->is_object()) for (auto& m : props->members()) pv.emplace_back(m.first, m.second);
            std::set<std::string> req;
            if (const jd::Value* r = get(schema, "required"); r && r->is_array()) for (auto& x : r->items()) if (x.is_string()) req.insert(x.as_string());
            const jd::Value* add_schema = nullptr;
            jd::Value tmp;
            if (addl) {
                if (addl->is_object()) add_schema = addl;
                else if (addl->is_bool() && addl->as_bool()) { tmp = jd::Value::object(); add_schema = &tmp; }
            }
            return add_rule(rule_name, build_object_rule(pv, req, name, add_schema));
        }
        if (type.empty() || type == "object") {
            if (const jd::Value* all = get(schema, "allOf"); all && all->is_array()) return visit_all_of(*all, name, rule_name);
        }
        const jd::Value* items = get(schema, "items");
        const jd::Value* prefix = get(schema, "prefixItems");
        if ((type.empty() || type == "array") && (items || prefix)) {
            if (prefix && prefix->is_array()) {
                std::string rule = "\"[\" space ";
                for (size_t i = 0; i < prefix->items().size(); ++i) {
                    if (i) rule += " \",\" space ";
                    rule += visit(prefix->items()[i], sub_name + "tuple-" + std::to_string(i));
                }
                rule += " space \"]\"";
                return add_rule(rule_name, rule);
            }
            int min_items = 0, max_items = std::numeric_limits<int>::max();
            if (const jd::Value* m = get(schema, "minItems"); m && m->is_number()) min_items = (int)m->as_int();
            if (const jd::Value* m = get(schema, "maxItems"); m && m->is_number()) max_items = (int)m->as_int();
            if (items && items->is_object() && items->members().empty() && min_items == 0 && max_items == std::numeric_limits<int>::max()) return visit_primitive(rule_name, "array");
            std::string item_rule = items ? visit(*items, sub_name + "item") : prim("value");
            return add_rule(rule_name, "\"[\" space " + build_repetition(item_rule, min_items, max_items, "\",\" space") + " space \"]\"");
        }
        if ((type.empty() || type == "string") && has(schema, "pattern")) return visit_pattern(get(schema, "pattern")->as_string(), rule_name);
        if ((type.empty() || type == "string") && has(schema, "format")) {
            std::string f = get(schema, "format")->as_string();
            if (f == "uuid" || (f.size() == 5 && f.compare(0, 4, "uuid") == 0 && f[4] >= '1' && f[4] <= '5')) return visit_primitive(rule_name, "uuid");
            if (f == "date" || f == "time" || f == "date-time") {
                std::string pn = f + "-string";
                return add_rule(rule_name, add_primitive(pn, format_rules().at(pn)));
            }
        }
        if (type == "string" && (has(schema, "minLength") || has(schema, "maxLength"))) {
            std::string char_rule = prim("char");
            int min_len = 0, max_len = std::numeric_limits<int>::max();
            if (const jd::Value* m = get(schema, "minLength"); m && m->is_number()) min_len = (int)m->as_int();
            if (const jd::Value* m = get(schema, "maxLength"); m && m->is_number()) max_len = (int)m->as_int();
            return add_rule(rule_name, "\"\\\"\" " + build_repetition(char_rule, min_len, max_len) + " \"\\\"\"");
        }
        if (type == "integer" && (has(schema, "minimum") || has(schema, "exclusiveMinimum") || has(schema, "maximum") || has(schema, "exclusiveMaximum"))) {
            int64_t lo = std::numeric_limits<int64_t>::min(), hi = std::numeric_limits<int64_t>::max();
            auto num = [&](const char* k, bool& found) -> double {
                const jd::Value* v = get(schema, k);
                found = v && v->is_number();
                return found ? v->as_double() : 0;
            };
            bool f;
            double v;
            v = num("minimum", f); if (f) lo = (int64_t)std::ceil(v);
            v = num("exclusiveMinimum", f); if (f) lo = std::max(lo == std::numeric_limits<int64_t>::min() ? lo : lo, (int64_t)std::floor(v) + 1);
            v = num("maximum", f); if (f) hi = (int64_t)std::floor(v);
            v = num("exclusiveMaximum", f); if (f) hi = std::min(hi, (int64_t)std::ceil(v) - 1);
            if (lo > hi) throw std::invalid_argument("integer range is empty");
            std::stringstream out;
            out << "(";
            build_min_max_int(lo, hi, out);
            out << ")";
            return add_rule(rule_name, out.str());
        }
        if (type == "object") return add_rule(rule_name, prim("object"));
        if (type == "array") return visit_primitive(rule_name, "array");
        if (type == "string") return visit_primitive(rule_name, "string");
        if (type == "integer") return visit_primitive(rule_name, "integer");
        if (type == "number") return visit_primitive(rule_name, "number");
        if (type == "boolean") return visit_primitive(rule_name, "boolean");
        if (type == "null") return visit_primitive(rule_name, "null");
        if (type.empty()) return add_rule(rule_name, prim("value"));
        throw std::invalid_argument("Unrecognized schema type: " + type);
    }

    std::string visit_all_of(const jd::Value& all, const std::string& name, const std::string& rule_name) {
        std::set<std::string> required;
        std::vector<std::pair<std::string, jd::Value>> properties;
        std::function<void(const jd::Value&, bool, int)> add_component = [&](const jd::Value& comp, bool is_required, int d) {
            if (d >= RECURSIVE_LIMIT) throw std::invalid_argument("JSON schema is nested too deep");
            if (const jd::Value* ref = get(comp, "$ref"); ref && ref->is_string()) {
                std::string frag;
                if (const jd::Value* t = resolve_pointer(ref->as_string(), frag)) add_component(*t, is_required, d + 1);
                return;
            }
            if (const jd::Value* p = get(comp, "properties"); p && p->is_object()) {
                for (auto& m : p->members()) {
                    properties.emplace_back(m.first, m.second);
                    if (is_required) required.insert(m.first);
                }
            }
            if (const jd::Value* r = get(comp, "required"); r && r->is_array() && is_required)
                for (auto& x : r->items()) if (x.is_string()) required.insert(x.as_string());
        };
        for (auto& c : all.items()) {
            if (const jd::Value* any = get(c, "anyOf"); any && any->is_array()) for (auto& a : any->items()) add_component(a, false, 0);
            else add_component(c, true, 0);
        }
        return add_rule(rule_name, build_object_rule(properties, required, name, nullptr));
    }
};

// ============================================================================ builder
SchemaGrammarBuilder::SchemaGrammarBuilder(SchemaOptions opt) : opt_(opt) {
    rules_["space"] = opt.compact ? "\"\"" : SPACE_RULE;
}
std::string SchemaGrammarBuilder::add_rule(const std::string& name, const std::string& rule) {
    std::string esc = gbnf_rule_name(name);
    if (!rules_.count(esc) || rules_[esc] == rule) { rules_[esc] = rule; return esc; }
    int i = 0;
    while (rules_.count(esc + std::to_string(i)) && rules_[esc + std::to_string(i)] != rule) ++i;
    std::string key = esc + std::to_string(i);
    rules_[key] = rule;
    return key;
}
std::string SchemaGrammarBuilder::add_schema(const std::string& name, const jd::Value& schema) {
    Conv c(*this, schema);
    std::string r = c.visit(schema, name == "root" ? "" : name);
    if (!c.errors.empty()) {
        std::string m = "JSON schema conversion failed:\n";
        for (auto& e : c.errors) m += e + "\n";
        throw std::invalid_argument(m);
    }
    return r;
}
std::string SchemaGrammarBuilder::add_primitive(const std::string& name) {
    Conv c(*this, jd::Value::object());
    auto it = primitive_rules().find(name);
    if (it == primitive_rules().end()) throw std::invalid_argument("unknown primitive " + name);
    return c.add_primitive(name, it->second);
}
std::string SchemaGrammarBuilder::build() const {
    std::string out;
    for (auto& kv : rules_) out += kv.first + " ::= " + kv.second + "\n";
    return out;
}

std::string json_schema_to_grammar(const jd::Value& schema, SchemaOptions opt) {
    SchemaGrammarBuilder b(opt);
    try {
        b.add_schema("", schema);
    } catch (const std::invalid_argument& e) {
        std::string m = e.what();
        if (m.rfind("JSON schema conversion failed", 0) == 0) throw;
        throw std::invalid_argument("JSON schema conversion failed:\n" + m);
    }
    return b.build();
}

std::string json_object_grammar() {
    SchemaGrammarBuilder b;
    b.add_primitive("object");
    b.add_rule("root", "object");
    return b.build();
}

}  // namespace qv::grammar
