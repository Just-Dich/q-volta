// Jinja2-compatible runtime values (Python object semantics: lists/dicts are shared references, str is code points).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qv::jd { class Value; }

namespace qv::jinja {

// Template syntax/runtime error (Python: TemplateError / UndefinedError / TypeError ...).
struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};
// Thrown by raise_exception(msg) inside a template; what() is the bare message.
struct RaisedError : Error {
    using Error::Error;
};

class Value;
struct Args;
using NativeFn = std::function<Value(Args&)>;

struct Obj {
    virtual ~Obj() = default;
};

class Value {
   public:
    enum class Kind : uint8_t { Undefined, None, Bool, Int, Float, Str, List, Tuple, Dict, Func, Loop };

    Value() = default;  // Undefined
    static Value none() { Value v; v.k_ = Kind::None; return v; }
    Value(bool b) : k_(Kind::Bool), i_(b) {}
    Value(int i) : k_(Kind::Int), i_(i) {}
    Value(long i) : k_(Kind::Int), i_(i) {}
    Value(long long i) : k_(Kind::Int), i_(i) {}
    Value(size_t i) : k_(Kind::Int), i_((int64_t)i) {}
    Value(double d) : k_(Kind::Float), d_(d) {}
    Value(const char* s) : k_(Kind::Str), s_(s) {}
    Value(std::string s) : k_(Kind::Str), s_(std::move(s)) {}
    Value(std::string_view s) : k_(Kind::Str), s_(s) {}
    static Value list(std::vector<Value> v = {});
    static Value tuple(std::vector<Value> v = {});
    static Value dict();
    static Value ns();  // jinja namespace() object (a dict whose attributes are assignable)
    static Value func(std::string name, NativeFn fn);
    static Value loop(std::shared_ptr<Obj> state);

    Kind kind() const { return k_; }
    bool is_undefined() const { return k_ == Kind::Undefined; }
    bool is_none() const { return k_ == Kind::None; }
    bool is_bool() const { return k_ == Kind::Bool; }
    bool is_int() const { return k_ == Kind::Int; }
    bool is_float() const { return k_ == Kind::Float; }
    bool is_number() const { return k_ == Kind::Int || k_ == Kind::Float; }  // bool counts as number in jinja; see tests
    bool is_string() const { return k_ == Kind::Str; }
    bool is_list() const { return k_ == Kind::List; }
    bool is_tuple() const { return k_ == Kind::Tuple; }
    bool is_seq() const { return k_ == Kind::List || k_ == Kind::Tuple; }
    bool is_dict() const { return k_ == Kind::Dict; }
    bool is_func() const { return k_ == Kind::Func; }
    bool is_ns() const { return k_ == Kind::Dict && ns_; }

    bool as_bool() const { return i_ != 0; }
    int64_t as_int() const { return k_ == Kind::Float ? (int64_t)d_ : i_; }
    double as_double() const { return k_ == Kind::Float ? d_ : (double)i_; }
    const std::string& as_string() const { return s_; }
    std::string& str_ref() { return s_; }

    std::vector<Value>& items();               // list/tuple elements
    const std::vector<Value>& items() const;
    struct DictData;
    DictData& dict_data();
    const DictData& dict_data() const;
    const Obj* obj() const { return o_.get(); }

    // dict helpers (string keys)
    const Value* find(const std::string& key) const;
    void set(const std::string& key, Value v);       // dict: insert or replace
    void set(const Value& key, Value v);
    void set(const char* key, Value v) { set(std::string(key), std::move(v)); }
    void push(Value v) { items().push_back(std::move(v)); }

    // Python semantics
    bool truthy() const;
    std::string str() const;    // Python str() (Undefined -> "")
    std::string repr() const;   // Python repr()
    size_t length() const;      // len()  (code points for str)
    static bool equals(const Value& a, const Value& b);
    static int compare(const Value& a, const Value& b);  // throws Error when unorderable
    const char* type_name() const;

   private:
    Kind k_ = Kind::Undefined;
    bool ns_ = false;
    int64_t i_ = 0;
    double d_ = 0;
    std::string s_;
    std::shared_ptr<Obj> o_;
};

struct Value::DictData : Obj {
    std::vector<std::pair<Value, Value>> kv;
    int find_idx(const Value& key) const;
};
struct ListData : Obj {
    std::vector<Value> v;
};
struct FuncData : Obj {
    std::string name;
    NativeFn fn;
};

struct Args {
    std::vector<Value> pos;
    std::vector<std::pair<std::string, Value>> kw;
    // positional index or keyword name
    const Value* get(size_t i, const char* name) const {
        if (i < pos.size()) return &pos[i];
        for (auto& p : kw) if (p.first == name) return &p.second;
        return nullptr;
    }
    const Value* kwarg(const char* name) const {
        for (auto& p : kw) if (p.first == name) return &p.second;
        return nullptr;
    }
};

// ---- text helpers (UTF-8 code points) ----
size_t u8_length(std::string_view s);
std::string u8_substr(std::string_view s, int64_t start_cp, int64_t count_cp);
std::vector<uint32_t> u8_decode(std::string_view s);
void u8_append(std::string& out, uint32_t cp);
std::string py_float_repr(double d);
std::string py_str_repr(std::string_view s);
// json.dumps equivalent (ensure_ascii=false unless asked). indent<0: compact-with-python-default separators.
struct JsonOpts {
    int indent = -1;
    std::string item_sep = ", ", key_sep = ": ";
    bool sort_keys = false, ensure_ascii = false;
};
std::string to_json_string(const Value& v, const JsonOpts& o);

// Conversion with the repo's JSON document model.
Value from_json(const jd::Value& j);
jd::Value to_json(const Value& v);

}  // namespace qv::jinja
