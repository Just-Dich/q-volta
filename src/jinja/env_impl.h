// Internal: shared implementation state and cross-file helpers of the jinja library.
#pragma once
#include <ctime>
#include <functional>
#include <unordered_map>

#include "jinja/ast.h"
#include "jinja/jinja.h"

namespace qv::jinja {

struct EnvImpl {
    EnvOptions opts;
    std::unordered_map<std::string, Value> globals;
    std::unordered_map<std::string, FilterFn> filters;
    std::unordered_map<std::string, TestFn> tests;
    std::function<std::time_t()> clock;
};

struct LoopStateBase : Obj {
    virtual Value attr(EnvImpl& env, const std::string& name) = 0;
    virtual Value call(EnvImpl& env, Args& a) = 0;
};

void register_builtins(EnvImpl& env);
Value call_value(EnvImpl& env, const Value& f, Args& a);
// bound method of a str/list/tuple/dict value, Undefined when there is none
Value bound_method(EnvImpl& env, const Value& self, const std::string& name);
Value apply_filter(EnvImpl& env, const std::string& name, const Value& v, Args& a);
bool apply_test(EnvImpl& env, const std::string& name, const Value& v, Args& a);
Value binop(Op op, const Value& a, const Value& b);
bool contains(const Value& container, const Value& item);
std::vector<Value> iterate(const Value& v);          // jinja `for` iteration (throws if not iterable)
Value get_attr(EnvImpl& env, const Value& obj, const std::string& name);
Value get_item(EnvImpl& env, const Value& obj, const Value& key);
Value slice_value(const Value& obj, const Value* start, const Value* stop, const Value* step);

}  // namespace qv::jinja
