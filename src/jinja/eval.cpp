// Tree-walking evaluator for the jinja AST + Environment/Template public API.
#include <algorithm>

#include "jinja/env_impl.h"

namespace qv::jinja {
namespace {

struct Scope {
    std::unordered_map<std::string, Value> vars;
    std::shared_ptr<Scope> parent;
    const Value* lookup(const std::string& n) const {
        for (const Scope* s = this; s; s = s->parent.get()) {
            auto it = s->vars.find(n);
            if (it != s->vars.end()) return &it->second;
        }
        return nullptr;
    }
};
using ScopeP = std::shared_ptr<Scope>;

enum class Flow { Normal, Break, Continue };

struct LoopState : LoopStateBase {
    std::vector<Value> items;
    size_t idx = 0;
    int depth = 0;  // 1-based
    std::function<Value(const Value&, int)> recurse;
    Value last_changed;
    bool changed_init = false;

    Value attr(EnvImpl&, const std::string& n) override {
        size_t len = items.size();
        if (n == "index0") return Value((long long)idx);
        if (n == "index") return Value((long long)idx + 1);
        if (n == "revindex0") return Value((long long)(len - idx - 1));
        if (n == "revindex") return Value((long long)(len - idx));
        if (n == "first") return Value(idx == 0);
        if (n == "last") return Value(idx + 1 == len);
        if (n == "length") return Value((long long)len);
        if (n == "depth") return Value((long long)depth);
        if (n == "depth0") return Value((long long)depth - 1);
        if (n == "previtem") return idx == 0 ? Value() : items[idx - 1];
        if (n == "nextitem") return idx + 1 >= len ? Value() : items[idx + 1];
        if (n == "cycle") {
            return Value::func("cycle", [this](Args& a) -> Value {
                if (a.pos.empty()) throw Error("no items for cycling given");
                return a.pos[idx % a.pos.size()];
            });
        }
        if (n == "changed") {
            return Value::func("changed", [this](Args& a) -> Value {
                Value cur = a.pos.size() == 1 ? a.pos[0] : Value::tuple(a.pos);
                if (!changed_init || !Value::equals(last_changed, cur)) { changed_init = true; last_changed = cur; return Value(true); }
                return Value(false);
            });
        }
        return Value();
    }
    Value call(EnvImpl&, Args& a) override {
        if (!recurse) throw Error("The loop was not declared as recursive");
        if (a.pos.size() != 1) throw Error("loop() takes exactly one argument");
        return recurse(a.pos[0], depth + 1);
    }
};

class Interp {
   public:
    Interp(EnvImpl& e) : env(e) {}
    EnvImpl& env;

    // ------------------------------------------------------------------ expressions
    Value eval(const Expr& e, const ScopeP& sc, const Value* injected = nullptr) {
        try {
            return eval_inner(e, sc, injected);
        } catch (RaisedError&) {
            throw;
        } catch (Error& ex) {
            std::string m = ex.what();
            if (m.find("(line ") == std::string::npos && e.line > 0) m += " (line " + std::to_string(e.line) + ")";
            throw Error(m);
        }
    }

    Value eval_inner(const Expr& e, const ScopeP& sc, const Value* injected) {
        using T = Expr::T;
        switch (e.t) {
            case T::Const: return e.cval;
            case T::Name: {
                const Value* v = sc->lookup(e.name);
                if (v) return *v;
                auto g = env.globals.find(e.name);
                if (g != env.globals.end()) return g->second;
                return Value();
            }
            case T::List: { std::vector<Value> v; for (auto& x : e.args) v.push_back(eval(*x, sc)); return Value::list(v); }
            case T::Tuple: { std::vector<Value> v; for (auto& x : e.args) v.push_back(eval(*x, sc)); return Value::tuple(v); }
            case T::Dict: {
                Value d = Value::dict();
                for (size_t i = 0; i + 1 < e.args.size(); i += 2) d.set(eval(*e.args[i], sc), eval(*e.args[i + 1], sc));
                return d;
            }
            case T::Attr: {
                Value o = eval(*e.a, sc);
                if (o.is_undefined()) throw Error("'" + e.name + "' lookup on undefined value");
                return get_attr(env, o, e.name);
            }
            case T::Item: {
                Value o = eval(*e.a, sc);
                if (e.b->t == T::Slice) {
                    Value s0, s1, s2;
                    bool h0 = (bool)e.b->a, h1 = (bool)e.b->b, h2 = (bool)e.b->c;
                    if (h0) s0 = eval(*e.b->a, sc);
                    if (h1) s1 = eval(*e.b->b, sc);
                    if (h2) s2 = eval(*e.b->c, sc);
                    if (o.is_undefined()) throw Error("undefined value cannot be sliced");
                    return slice_value(o, h0 ? &s0 : nullptr, h1 ? &s1 : nullptr, h2 ? &s2 : nullptr);
                }
                Value k = eval(*e.b, sc);
                return get_item(env, o, k);
            }
            case T::Slice: throw Error("unexpected slice");
            case T::Call: {
                Value f = eval(*e.a, sc);
                Args a;
                for (auto& x : e.args) a.pos.push_back(eval(*x, sc));
                for (auto& k : e.kwargs) a.kw.emplace_back(k.first, eval(*k.second, sc));
                if (f.is_undefined()) throw Error("'" + describe(*e.a) + "' is undefined");
                return call_value(env, f, a);
            }
            case T::Filter: {
                Value in = e.a ? eval(*e.a, sc, injected) : (injected ? *injected : Value());
                Args a;
                for (auto& x : e.args) a.pos.push_back(eval(*x, sc));
                for (auto& k : e.kwargs) a.kw.emplace_back(k.first, eval(*k.second, sc));
                return apply_filter(env, e.name, in, a);
            }
            case T::Test: {
                Value in = eval(*e.a, sc);
                Args a;
                for (auto& x : e.args) a.pos.push_back(eval(*x, sc));
                for (auto& k : e.kwargs) a.kw.emplace_back(k.first, eval(*k.second, sc));
                bool r = apply_test(env, e.name, in, a);
                return Value(e.negate ? !r : r);
            }
            case T::Neg: {
                Value v = eval(*e.a, sc);
                if (v.is_float()) return Value(-v.as_double());
                if (v.is_int() || v.is_bool()) return Value((long long)-v.as_int());
                throw Error(std::string("bad operand type for unary -: '") + v.type_name() + "'");
            }
            case T::Pos: {
                Value v = eval(*e.a, sc);
                if (v.is_float()) return v;
                if (v.is_int() || v.is_bool()) return Value((long long)v.as_int());
                throw Error(std::string("bad operand type for unary +: '") + v.type_name() + "'");
            }
            case T::Not: return Value(!eval(*e.a, sc).truthy());
            case T::Bin: {
                Value a = eval(*e.a, sc);
                Value b = eval(*e.b, sc);
                return binop(e.op, a, b);
            }
            case T::Compare: {
                Value l = eval(*e.a, sc);
                for (auto& c : e.cmp) {
                    Value r = eval(*c.second, sc);
                    if (!binop(c.first, l, r).as_bool()) return Value(false);
                    l = r;
                }
                return Value(true);
            }
            case T::And: { Value l = eval(*e.a, sc); return l.truthy() ? eval(*e.b, sc) : l; }
            case T::Or: { Value l = eval(*e.a, sc); return l.truthy() ? l : eval(*e.b, sc); }
            case T::Cond: {
                if (eval(*e.b, sc).truthy()) return eval(*e.a, sc);
                return e.c ? eval(*e.c, sc) : Value();
            }
        }
        return Value();
    }

    static std::string describe(const Expr& e) {
        if (e.t == Expr::T::Name) return e.name;
        if (e.t == Expr::T::Attr) return describe(*e.a) + "." + e.name;
        return "expression";
    }

    // ------------------------------------------------------------------ assignment
    void assign(const Expr& target, const Value& v, const ScopeP& sc) {
        if (target.t == Expr::T::Name) { sc->vars[target.name] = v; return; }
        if (target.t == Expr::T::Tuple) {
            auto items = iterate(v);
            if (items.size() != target.args.size())
                throw Error(items.size() < target.args.size() ? "not enough values to unpack (expected " + std::to_string(target.args.size()) + ", got " + std::to_string(items.size()) + ")"
                                                              : "too many values to unpack (expected " + std::to_string(target.args.size()) + ")");
            for (size_t i = 0; i < items.size(); ++i) assign(*target.args[i], items[i], sc);
            return;
        }
        if (target.t == Expr::T::Attr) {
            Value o = eval(*target.a, sc);
            if (!o.is_ns()) throw Error("cannot assign attribute on non-namespace object");
            o.set(target.name, v);
            return;
        }
        throw Error("cannot assign to expression");
    }

    // ------------------------------------------------------------------ statements
    Flow exec_body(const std::vector<StmtP>& body, const ScopeP& sc, std::string& out) {
        for (auto& s : body) {
            Flow f = exec(*s, sc, out);
            if (f != Flow::Normal) return f;
        }
        return Flow::Normal;
    }

    Flow exec(const Stmt& s, const ScopeP& sc, std::string& out) {
        using T = Stmt::T;
        try {
            switch (s.t) {
                case T::Data: out += s.text; return Flow::Normal;
                case T::Output: out += eval(*s.expr, sc).str(); return Flow::Normal;
                case T::If:
                    if (eval(*s.expr, sc).truthy()) return exec_body(s.body, sc, out);
                    return exec_body(s.else_body, sc, out);
                case T::Seq: return exec_body(s.body, sc, out);
                case T::For: return exec_for(s, sc, out, nullptr, 1);
                case T::Set: assign(*s.target, eval(*s.expr, sc), sc); return Flow::Normal;
                case T::SetBlock: {
                    std::string buf;
                    exec_body(s.body, sc, buf);
                    Value v(buf);
                    if (s.expr) v = eval(*s.expr, sc, &v);
                    assign(*s.target, v, sc);
                    return Flow::Normal;
                }
                case T::Macro: sc->vars[s.text] = make_macro(s, sc, nullptr); return Flow::Normal;
                case T::CallBlock: {
                    const Expr& call = *s.expr;
                    Value f = eval(*call.a, sc);
                    Args a;
                    for (auto& x : call.args) a.pos.push_back(eval(*x, sc));
                    for (auto& k : call.kwargs) a.kw.emplace_back(k.first, eval(*k.second, sc));
                    a.kw.emplace_back("caller", make_macro(s, sc, "caller"));
                    out += call_value(env, f, a).str();
                    return Flow::Normal;
                }
                case T::FilterBlock: {
                    std::string buf;
                    exec_body(s.body, sc, buf);
                    Value v(buf);
                    out += eval(*s.expr, sc, &v).str();
                    return Flow::Normal;
                }
                case T::With: {
                    auto child = std::make_shared<Scope>();
                    child->parent = sc;
                    for (auto& p : s.params) child->vars[p.name] = eval(*p.def, sc);
                    return exec_body(s.body, child, out);
                }
                case T::Do: eval(*s.expr, sc); return Flow::Normal;
                case T::Break: return Flow::Break;
                case T::Continue: return Flow::Continue;
            }
        } catch (RaisedError&) {
            throw;
        } catch (Error& ex) {
            std::string m = ex.what();
            if (m.find("(line ") == std::string::npos) m += " (line " + std::to_string(s.line) + ")";
            throw Error(m);
        }
        return Flow::Normal;
    }

    Flow exec_for(const Stmt& s, const ScopeP& sc, std::string& out, const Value* recursive_iter, int depth) {
        Value iterable = recursive_iter ? *recursive_iter : eval(*s.expr, sc);
        if (iterable.is_none()) throw Error("'NoneType' object is not iterable");
        std::vector<Value> items = iterate(iterable);
        if (s.cond) {
            std::vector<Value> kept;
            for (auto& it : items) {
                auto tmp = std::make_shared<Scope>();
                tmp->parent = sc;
                assign(*s.target, it, tmp);
                if (eval(*s.cond, tmp).truthy()) kept.push_back(it);
            }
            items = std::move(kept);
        }
        if (items.empty()) {
            exec_body(s.else_body, sc, out);
            return Flow::Normal;
        }
        auto st = std::make_shared<LoopState>();
        st->items = std::move(items);
        st->depth = depth;
        if (s.recursive) {
            const Stmt* sp = &s;
            ScopeP scp = sc;
            st->recurse = [this, sp, scp](const Value& it, int d) -> Value {
                std::string buf;
                exec_for(*sp, scp, buf, &it, d);
                return Value(buf);
            };
        }
        Value loopv = Value::loop(st);
        for (size_t i = 0; i < st->items.size(); ++i) {
            st->idx = i;
            auto child = std::make_shared<Scope>();
            child->parent = sc;
            child->vars["loop"] = loopv;
            assign(*s.target, st->items[i], child);
            Flow f = exec_body(s.body, child, out);
            if (f == Flow::Break) break;
        }
        return Flow::Normal;
    }

    // macro / caller closure
    Value make_macro(const Stmt& s, const ScopeP& def_scope, const char* kind) {
        std::weak_ptr<Scope> wscope = def_scope;
        const Stmt* sp = &s;
        std::string name = kind ? "caller" : s.text;
        return Value::func(name, [this, sp, wscope, name](Args& a) -> Value {
            ScopeP def = wscope.lock();
            if (!def) throw Error("macro '" + name + "' called outside of its defining scope");
            auto sc = std::make_shared<Scope>();
            sc->parent = def;
            const auto& params = sp->params;
            std::vector<Value> extra;
            for (size_t i = 0; i < a.pos.size(); ++i) {
                if (i < params.size()) sc->vars[params[i].name] = a.pos[i];
                else extra.push_back(a.pos[i]);
            }
            Value kwextra = Value::dict();
            for (auto& k : a.kw) {
                bool declared = std::any_of(params.begin(), params.end(), [&](const Param& p) { return p.name == k.first; });
                if (declared) sc->vars[k.first] = k.second;
                else if (k.first == "caller") sc->vars["caller"] = k.second;
                else kwextra.set(k.first, k.second);
            }
            for (size_t i = 0; i < params.size(); ++i) {
                if (sc->vars.count(params[i].name)) continue;
                sc->vars[params[i].name] = params[i].def ? eval(*params[i].def, sc) : Value();
            }
            sc->vars["varargs"] = Value::list(extra);
            sc->vars["kwargs"] = kwextra;
            std::string buf;
            exec_body(sp->body, sc, buf);
            return Value(buf);
        });
    }
};

}  // namespace

// ============================================================================ public API
Environment::Environment(EnvOptions opt) : impl_(std::make_shared<EnvImpl>()) {
    impl_->opts = opt;
    register_builtins(*impl_);
}
void Environment::set_global(const std::string& n, Value v) { impl_->globals[n] = std::move(v); }
void Environment::add_filter(const std::string& n, FilterFn f) { impl_->filters[n] = std::move(f); }
void Environment::add_test(const std::string& n, TestFn f) { impl_->tests[n] = std::move(f); }
void Environment::set_clock(std::function<std::time_t()> now) { impl_->clock = std::move(now); }

Template Environment::compile(std::string_view source, const std::string& name) const {
    ParseOptions po;
    po.trim_blocks = impl_->opts.trim_blocks;
    po.lstrip_blocks = impl_->opts.lstrip_blocks;
    po.keep_trailing_newline = impl_->opts.keep_trailing_newline;
    Template t;
    t.env_ = impl_;
    t.ast_ = parse_template(source, name, po);
    t.name_ = name;
    return t;
}

std::string Template::render(const Value& ctx) const {
    if (!ast_) throw Error("template not compiled");
    Interp in(*env_);
    auto root = std::make_shared<Scope>();
    if (ctx.is_dict())
        for (auto& p : ctx.dict_data().kv) root->vars[p.first.str()] = p.second;
    std::string out;
    in.exec_body(ast_->body, root, out);
    return out;
}

}  // namespace qv::jinja
