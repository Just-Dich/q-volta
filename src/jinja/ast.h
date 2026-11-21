// Internal: template AST (shared by parse.cpp and eval.cpp).
#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "jinja/value.h"

namespace qv::jinja {

struct Expr;
struct Stmt;
using ExprP = std::shared_ptr<Expr>;
using StmtP = std::shared_ptr<Stmt>;

enum class Op { Add, Sub, Mul, Div, FloorDiv, Mod, Pow, Concat, Eq, Ne, Lt, Le, Gt, Ge, In, NotIn };

struct Expr {
    enum class T { Const, Name, List, Tuple, Dict, Attr, Item, Slice, Call, Filter, Test, Neg, Pos, Not, Bin, Compare, And, Or, Cond };
    T t = T::Const;
    int line = 0;
    Value cval;
    std::string name;
    ExprP a, b, c;
    std::vector<ExprP> args;                              // List/Tuple elems, Dict k,v,k,v..., call args
    std::vector<std::pair<std::string, ExprP>> kwargs;    // Call/Filter/Test keyword args
    Op op = Op::Add;                                      // Bin
    std::vector<std::pair<Op, ExprP>> cmp;                // Compare chain after a
    bool negate = false;                                  // Test: "is not"
};

struct Param {
    std::string name;
    ExprP def;
};

struct Stmt {
    enum class T { Data, Output, If, For, Set, SetBlock, Macro, CallBlock, FilterBlock, Break, Continue, With, Do, Seq };
    T t = T::Data;
    int line = 0;
    std::string text;                 // Data; Macro/SetBlock name
    ExprP expr;                       // Output expr, If cond, For iterable, Set value, Do expr, CallBlock call, FilterBlock filter chain
    ExprP target;                     // Set target (Name/Tuple/Attr)
    ExprP cond;                       // For ... if cond
    std::vector<std::string> names;   // For targets
    std::vector<Param> params;        // Macro / CallBlock caller params / With assignments
    bool recursive = false;
    std::vector<StmtP> body, else_body;
};

struct TemplateAst {
    std::vector<StmtP> body;
    std::string name;
};

struct ParseOptions {
    bool trim_blocks = true, lstrip_blocks = true, keep_trailing_newline = false;
};

std::shared_ptr<TemplateAst> parse_template(std::string_view src, const std::string& name, const ParseOptions& opt);

}  // namespace qv::jinja
