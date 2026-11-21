// Jinja lexer (with trim_blocks / lstrip_blocks / '-' whitespace control, exactly as jinja2's lexer) and parser.
#include <cmath>
#include <cstring>
#include <set>

#include "jinja/ast.h"

namespace qv::jinja {
namespace {

enum class TT { Data, VarBegin, VarEnd, BlockBegin, BlockEnd, Name, Int, Float, Str, Op, End };
struct Token {
    TT t = TT::End;
    std::string s;
    int64_t i = 0;
    double d = 0;
    int line = 1;
};

[[noreturn]] void fail(const std::string& m, int line) {
    throw Error(m + " (line " + std::to_string(line) + ")");
}

bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }
bool is_name_start(unsigned char c) { return isalpha(c) || c == '_' || c >= 0x80; }
bool is_name_char(unsigned char c) { return isalnum(c) || c == '_' || c >= 0x80; }

std::string unescape(const std::string& raw, int line) {
    std::string o;
    for (size_t i = 0; i < raw.size(); ++i) {
        char c = raw[i];
        if (c != '\\' || i + 1 >= raw.size()) { o += c; continue; }
        char e = raw[++i];
        switch (e) {
            case 'n': o += '\n'; break;
            case 't': o += '\t'; break;
            case 'r': o += '\r'; break;
            case 'a': o += '\a'; break;
            case 'b': o += '\b'; break;
            case 'f': o += '\f'; break;
            case 'v': o += '\v'; break;
            case '\\': o += '\\'; break;
            case '\'': o += '\''; break;
            case '"': o += '"'; break;
            case '\n': break;
            case 'x': case 'u': case 'U': {
                size_t n = e == 'x' ? 2 : e == 'u' ? 4 : 8;
                if (i + n < raw.size() + 0 && i + n <= raw.size() - 0) {
                    std::string h = raw.substr(i + 1, n);
                    if (h.size() == n && h.find_first_not_of("0123456789abcdefABCDEF") == std::string::npos) {
                        uint32_t cp = (uint32_t)strtoul(h.c_str(), nullptr, 16);
                        i += n;
                        if (cp >= 0xD800 && cp < 0xDC00 && raw.compare(i + 1, 2, "\\u") == 0 && i + 6 < raw.size() + 1) {
                            std::string h2 = raw.substr(i + 3, 4);
                            uint32_t lo = (uint32_t)strtoul(h2.c_str(), nullptr, 16);
                            if (lo >= 0xDC00 && lo < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); i += 6; }
                        }
                        u8_append(o, cp);
                        break;
                    }
                }
                o += '\\'; o += e;
                break;
            }
            default:
                if (e >= '0' && e <= '7') {
                    int v = 0, n = 0;
                    while (n < 3 && i < raw.size() && raw[i] >= '0' && raw[i] <= '7') { v = v * 8 + (raw[i] - '0'); ++i; ++n; }
                    --i;
                    u8_append(o, (uint32_t)v);
                } else { o += '\\'; o += e; }
        }
    }
    (void)line;
    return o;
}

class Lexer {
   public:
    Lexer(std::string_view src, const ParseOptions& o) : s_(src), o_(o) {}

    std::vector<Token> run() {
        std::string src(s_);
        if (!o_.keep_trailing_newline) {
            if (src.size() >= 2 && src.compare(src.size() - 2, 2, "\r\n") == 0) src.resize(src.size() - 2);
            else if (!src.empty() && (src.back() == '\n' || src.back() == '\r')) src.pop_back();
        }
        s_ = src;
        std::vector<Token> out;
        bool line_starting = true;
        size_t pos = 0;
        int line = 1;
        auto count_lines = [&](size_t a, size_t b) { for (size_t k = a; k < b; ++k) if (s_[k] == '\n') ++line; };
        while (pos <= s_.size()) {
            // find next tag start
            size_t tag = std::string::npos;
            for (size_t k = pos; k + 1 < s_.size(); ++k) {
                if (s_[k] == '{' && (s_[k + 1] == '{' || s_[k + 1] == '%' || s_[k + 1] == '#')) { tag = k; break; }
            }
            if (tag == std::string::npos) {
                if (pos < s_.size()) push(out, TT::Data, s_.substr(pos), line);
                break;
            }
            char kind = s_[tag + 1];
            std::string data = s_.substr(pos, tag - pos);
            char mark = tag + 2 < s_.size() ? s_[tag + 2] : 0;
            if (mark == '-') {
                size_t e = data.size();
                while (e > 0 && is_ws(data[e - 1])) --e;
                data.resize(e);
            } else if (o_.lstrip_blocks && mark != '+' && kind != '{') {
                size_t l_pos = data.rfind('\n');
                l_pos = l_pos == std::string::npos ? 0 : l_pos + 1;
                if (l_pos > 0 || line_starting) {
                    bool all_ws = true;
                    for (size_t k = l_pos; k < data.size(); ++k) if (data[k] != ' ' && data[k] != '\t') all_ws = false;
                    if (all_ws) data.resize(l_pos);
                }
            }
            if (!data.empty()) push(out, TT::Data, data, line);
            count_lines(pos, tag);
            size_t p = tag + 2;
            if (mark == '-' || mark == '+') ++p;
            if (kind == '#') {
                size_t e = s_.find("#}", p);
                if (e == std::string::npos) fail("Missing end of comment tag", line);
                // '-#}' strips following whitespace
                bool dash = e > p && s_[e - 1] == '-';
                count_lines(p, e);
                p = e + 2;
                if (dash) { while (p < s_.size() && is_ws(s_[p])) { if (s_[p] == '\n') ++line; ++p; } line_starting = p > 0 && s_[p - 1] == '\n'; }
                else if (o_.trim_blocks) { if (p < s_.size() && s_[p] == '\n') { ++p; ++line; line_starting = true; } else if (s_.compare(p, 2, "\r\n") == 0) { p += 2; ++line; line_starting = true; } else line_starting = false; }
                else line_starting = false;
                pos = p;
                continue;
            }
            bool is_var = kind == '{';
            push(out, is_var ? TT::VarBegin : TT::BlockBegin, "", line);
            // tokenize inside tag
            std::vector<char> stack;
            bool closed = false;
            while (p < s_.size()) {
                char c = s_[p];
                if (is_ws(c)) { if (c == '\n') ++line; ++p; continue; }
                // end markers
                if (stack.empty()) {
                    const char* endm = is_var ? "}}" : "%}";
                    bool dash = c == '-' && s_.compare(p + 1, 2, endm) == 0;
                    bool plus = !is_var && c == '+' && s_.compare(p + 1, 2, endm) == 0;
                    if (dash || plus || s_.compare(p, 2, endm) == 0) {
                        size_t q = p + ((dash || plus) ? 3 : 2);
                        push(out, is_var ? TT::VarEnd : TT::BlockEnd, "", line);
                        if (dash) {
                            while (q < s_.size() && is_ws(s_[q])) { if (s_[q] == '\n') ++line; ++q; }
                            line_starting = q > 0 && s_[q - 1] == '\n';
                        } else if (!is_var && o_.trim_blocks && !plus) {
                            if (q < s_.size() && s_[q] == '\n') { ++q; ++line; line_starting = true; }
                            else if (s_.compare(q, 2, "\r\n") == 0) { q += 2; ++line; line_starting = true; }
                            else line_starting = false;
                        } else line_starting = false;
                        pos = q;
                        closed = true;
                        // raw block
                        if (!is_var && out.size() >= 3 && out[out.size() - 2].t == TT::Name && out[out.size() - 2].s == "raw" &&
                            out[out.size() - 3].t == TT::BlockBegin) {
                            out.pop_back(); out.pop_back(); out.pop_back();  // drop {% raw %}
                            size_t k = pos;
                            size_t found = std::string::npos, endp = 0;
                            while (k < s_.size()) {
                                size_t b = s_.find("{%", k);
                                if (b == std::string::npos) break;
                                size_t m = b + 2;
                                if (m < s_.size() && (s_[m] == '-' || s_[m] == '+')) ++m;
                                while (m < s_.size() && is_ws(s_[m])) ++m;
                                if (s_.compare(m, 6, "endraw") == 0) {
                                    size_t n = m + 6;
                                    while (n < s_.size() && is_ws(s_[n])) ++n;
                                    bool d2 = false;
                                    if (n < s_.size() && (s_[n] == '-')) { d2 = true; ++n; }
                                    if (s_.compare(n, 2, "%}") == 0) {
                                        found = b; endp = n + 2;
                                        (void)d2;
                                        break;
                                    }
                                }
                                k = b + 2;
                            }
                            if (found == std::string::npos) fail("Missing end of raw directive", line);
                            std::string body = s_.substr(pos, found - pos);
                            size_t m = found + 2;
                            if (m < s_.size() && s_[m] == '-') {
                                size_t e = body.size();
                                while (e > 0 && is_ws(body[e - 1])) --e;
                                body.resize(e);
                            }
                            if (!body.empty()) push(out, TT::Data, body, line);
                            count_lines(pos, endp);
                            // whitespace control after endraw
                            size_t q2 = endp;
                            bool dash2 = s_[endp - 3] == '-';
                            if (dash2) { while (q2 < s_.size() && is_ws(s_[q2])) ++q2; }
                            else if (o_.trim_blocks && q2 < s_.size() && s_[q2] == '\n') ++q2;
                            pos = q2;
                        }
                        break;
                    }
                }
                lex_token(out, p, line, stack);
            }
            if (!closed) fail(is_var ? "Unexpected end of template, expected '}}'" : "Unexpected end of template, expected '%}'", line);
            if (pos >= s_.size()) break;
        }
        push(out, TT::End, "", line);
        return out;
    }

   private:
    std::string s_;
    const ParseOptions& o_;
    void push(std::vector<Token>& v, TT t, std::string s, int line) {
        Token k;
        k.t = t; k.s = std::move(s); k.line = line;
        v.push_back(std::move(k));
    }
    void lex_token(std::vector<Token>& out, size_t& p, int& line, std::vector<char>& stack) {
        unsigned char c = s_[p];
        if (is_name_start(c)) {
            size_t q = p;
            while (q < s_.size() && is_name_char(s_[q])) ++q;
            push(out, TT::Name, s_.substr(p, q - p), line);
            p = q;
            return;
        }
        if (isdigit(c)) {
            size_t q = p;
            std::string digits;
            while (q < s_.size() && (isdigit((unsigned char)s_[q]) || s_[q] == '_')) { if (s_[q] != '_') digits += s_[q]; ++q; }
            bool flt = false;
            if (q + 1 < s_.size() && s_[q] == '.' && isdigit((unsigned char)s_[q + 1])) {
                flt = true;
                digits += '.';
                ++q;
                while (q < s_.size() && (isdigit((unsigned char)s_[q]) || s_[q] == '_')) { if (s_[q] != '_') digits += s_[q]; ++q; }
            }
            if (q < s_.size() && (s_[q] == 'e' || s_[q] == 'E')) {
                size_t r = q + 1;
                if (r < s_.size() && (s_[r] == '+' || s_[r] == '-')) ++r;
                if (r < s_.size() && isdigit((unsigned char)s_[r])) {
                    flt = true;
                    digits += s_.substr(q, r - q);
                    while (r < s_.size() && isdigit((unsigned char)s_[r])) digits += s_[r++];
                    q = r;
                }
            }
            Token k;
            k.line = line;
            if (flt) { k.t = TT::Float; k.d = strtod(digits.c_str(), nullptr); }
            else { k.t = TT::Int; k.i = (int64_t)strtoull(digits.c_str(), nullptr, 10); }
            out.push_back(k);
            p = q;
            return;
        }
        if (c == '"' || c == '\'') {
            size_t q = p + 1;
            std::string raw;
            while (q < s_.size() && s_[q] != (char)c) {
                if (s_[q] == '\\' && q + 1 < s_.size()) { raw += s_[q]; ++q; }
                if (s_[q] == '\n') ++line;
                raw += s_[q++];
            }
            if (q >= s_.size()) fail("Unterminated string literal", line);
            push(out, TT::Str, unescape(raw, line), line);
            p = q + 1;
            return;
        }
        static const char* ops3[] = {"//", "**", "==", "!=", ">=", "<="};
        for (auto o : ops3)
            if (s_.compare(p, 2, o) == 0) { push(out, TT::Op, o, line); p += 2; return; }
        if (strchr("+-*/%=<>!~|.,:()[]{}", c) && c != '!') {
            if (c == '(' || c == '[' || c == '{') stack.push_back((char)c);
            else if (c == ')' || c == ']' || c == '}') {
                if (stack.empty()) fail(std::string("unexpected '") + (char)c + "'", line);
                char open = stack.back();
                if ((c == ')' && open != '(') || (c == ']' && open != '[') || (c == '}' && open != '{'))
                    fail(std::string("unexpected '") + (char)c + "', expected closing of '" + open + "'", line);
                stack.pop_back();
            }
            push(out, TT::Op, std::string(1, (char)c), line);
            ++p;
            return;
        }
        fail(std::string("unexpected char '") + (char)c + "'", line);
    }
};

// ----------------------------------------------------------------------------------- parser
class Parser {
   public:
    Parser(std::vector<Token> t) : t_(std::move(t)) {}

    std::vector<StmtP> parse_all() {
        auto b = parse_body({});
        if (cur().t != TT::End) fail("Encountered unknown tag '" + peek_tag() + "'", cur().line);
        return b;
    }

   private:
    std::vector<Token> t_;
    size_t p_ = 0;

    const Token& cur() const { return t_[p_]; }
    const Token& at(size_t k) const { return t_[std::min(p_ + k, t_.size() - 1)]; }
    Token next() { return t_[p_++ < t_.size() - 1 ? p_ - 1 : t_.size() - 1]; }
    bool is_op(const char* o, size_t k = 0) const { return at(k).t == TT::Op && at(k).s == o; }
    bool is_name(const char* n, size_t k = 0) const { return at(k).t == TT::Name && at(k).s == n; }
    bool skip_op(const char* o) { if (is_op(o)) { ++p_; return true; } return false; }
    bool skip_name(const char* n) { if (is_name(n)) { ++p_; return true; } return false; }
    void expect_op(const char* o) {
        if (!is_op(o)) fail(std::string("expected '") + o + "', got '" + describe() + "'", cur().line);
        ++p_;
    }
    std::string expect_name() {
        if (cur().t != TT::Name) fail("expected name, got '" + describe() + "'", cur().line);
        return next().s;
    }
    std::string describe() const {
        switch (cur().t) {
            case TT::End: return "end of template";
            case TT::BlockEnd: return "%}";
            case TT::VarEnd: return "}}";
            case TT::Int: return std::to_string(cur().i);
            default: return cur().s;
        }
    }
    std::string peek_tag() const { return (cur().t == TT::BlockBegin && at(1).t == TT::Name) ? at(1).s : describe(); }

    ExprP mk(Expr::T t, int line) {
        auto e = std::make_shared<Expr>();
        e->t = t;
        e->line = line;
        return e;
    }
    StmtP mks(Stmt::T t, int line) {
        auto s = std::make_shared<Stmt>();
        s->t = t;
        s->line = line;
        return s;
    }

    // -------- statements
    std::vector<StmtP> parse_body(const std::set<std::string>& ends) {
        std::vector<StmtP> out;
        while (cur().t != TT::End) {
            const Token& t = cur();
            if (t.t == TT::Data) {
                auto s = mks(Stmt::T::Data, t.line);
                s->text = t.s;
                out.push_back(s);
                ++p_;
            } else if (t.t == TT::VarBegin) {
                ++p_;
                auto s = mks(Stmt::T::Output, t.line);
                s->expr = parse_expression();
                if (cur().t != TT::VarEnd) fail("expected '}}', got '" + describe() + "'", cur().line);
                ++p_;
                out.push_back(s);
            } else if (t.t == TT::BlockBegin) {
                if (at(1).t != TT::Name) fail("expected tag name", t.line);
                const std::string& tag = at(1).s;
                if (ends.count(tag)) return out;
                p_ += 2;
                out.push_back(parse_tag(tag, t.line));
            } else {
                fail("unexpected token '" + describe() + "'", t.line);
            }
        }
        if (!ends.empty()) {
            std::string e;
            for (auto& x : ends) e += (e.empty() ? "" : " or ") + std::string("'") + x + "'";
            fail("Unexpected end of template. Jinja was looking for the following tags: " + e, cur().line);
        }
        return out;
    }
    void end_block() {
        if (cur().t != TT::BlockEnd) fail("expected '%}', got '" + describe() + "'", cur().line);
        ++p_;
    }
    void consume_end_tag(const char* name) {  // at BlockBegin Name(name)
        p_ += 2;
        (void)name;
        end_block();
    }

    StmtP parse_tag(const std::string& tag, int line) {
        if (tag == "if") return parse_if(line);
        if (tag == "for") return parse_for(line);
        if (tag == "set") return parse_set(line);
        if (tag == "macro") return parse_macro(line);
        if (tag == "call") return parse_call(line);
        if (tag == "filter") return parse_filter_block(line);
        if (tag == "break" || tag == "continue") {
            end_block();
            return mks(tag == "break" ? Stmt::T::Break : Stmt::T::Continue, line);
        }
        if (tag == "do") {
            auto s = mks(Stmt::T::Do, line);
            s->expr = parse_expression();
            end_block();
            return s;
        }
        if (tag == "with") {
            auto s = mks(Stmt::T::With, line);
            while (cur().t != TT::BlockEnd) {
                Param p;
                p.name = expect_name();
                expect_op("=");
                p.def = parse_expression();
                s->params.push_back(p);
                if (!skip_op(",")) break;
            }
            end_block();
            s->body = parse_body({"endwith"});
            consume_end_tag("endwith");
            return s;
        }
        if (tag == "generation" || tag == "autoescape") {
            if (tag == "autoescape") parse_expression();
            end_block();
            std::string endt = "end" + tag;
            auto s = mks(Stmt::T::Seq, line);
            s->body = parse_body({endt});
            consume_end_tag(endt.c_str());
            return s;
        }
        fail("Encountered unknown tag '" + tag + "'", line);
    }

    StmtP parse_if(int line) {
        auto s = mks(Stmt::T::If, line);
        s->expr = parse_expression();
        end_block();
        s->body = parse_body({"elif", "else", "endif"});
        if (is_name("elif", 1)) {
            int l2 = cur().line;
            p_ += 2;
            s->else_body.push_back(parse_if(l2));  // nested chain consumes the endif
            return s;
        }
        if (is_name("else", 1)) {
            p_ += 2;
            end_block();
            s->else_body = parse_body({"endif"});
        }
        consume_end_tag("endif");
        return s;
    }

    ExprP parse_target() {
        // Name | Name, Name | (Name, Name) | name.attr
        int line = cur().line;
        if (skip_op("(")) {
            auto t = mk(Expr::T::Tuple, line);
            while (!is_op(")")) {
                t->args.push_back(parse_target());
                if (!skip_op(",")) break;
            }
            expect_op(")");
            return t;
        }
        auto n = mk(Expr::T::Name, line);
        n->name = expect_name();
        ExprP e = n;
        while (is_op(".")) {
            ++p_;
            auto a = mk(Expr::T::Attr, line);
            a->a = e;
            a->name = expect_name();
            e = a;
        }
        return e;
    }

    StmtP parse_for(int line) {
        auto s = mks(Stmt::T::For, line);
        ExprP tgt;
        {
            std::vector<ExprP> ts;
            bool tup = false;
            for (;;) {
                ts.push_back(parse_target());
                if (skip_op(",")) { tup = true; if (is_name("in")) break; continue; }
                break;
            }
            if (tup || ts.size() > 1) { auto t = mk(Expr::T::Tuple, line); t->args = ts; tgt = t; }
            else tgt = ts[0];
        }
        s->target = tgt;
        if (!skip_name("in")) fail("expected 'in' in for loop", cur().line);
        s->expr = parse_or();
        if (skip_name("if")) s->cond = parse_or();
        if (skip_name("recursive")) s->recursive = true;
        end_block();
        s->body = parse_body({"else", "endfor"});
        if (is_name("else", 1)) {
            p_ += 2;
            end_block();
            s->else_body = parse_body({"endfor"});
        }
        consume_end_tag("endfor");
        return s;
    }

    StmtP parse_set(int line) {
        auto tgt = parse_target();
        // tuple target without parens: a, b = ...
        if (is_op(",")) {
            auto t = mk(Expr::T::Tuple, line);
            t->args.push_back(tgt);
            while (skip_op(",")) t->args.push_back(parse_target());
            tgt = t;
        }
        if (skip_op("=")) {
            auto s = mks(Stmt::T::Set, line);
            s->target = tgt;
            ExprP v = parse_expression();
            if (is_op(",")) {
                auto t = mk(Expr::T::Tuple, line);
                t->args.push_back(v);
                while (skip_op(",")) { if (cur().t == TT::BlockEnd) break; t->args.push_back(parse_expression()); }
                v = t;
            }
            s->expr = v;
            end_block();
            return s;
        }
        auto s = mks(Stmt::T::SetBlock, line);
        s->target = tgt;
        if (is_op("|")) {
            // filter chain applied to the captured body
            ExprP in = nullptr;
            ExprP chain = in;
            while (skip_op("|")) {
                auto f = mk(Expr::T::Filter, cur().line);
                f->name = expect_name();
                f->a = chain;
                if (is_op("(")) parse_call_args(f->args, f->kwargs);
                chain = f;
            }
            s->expr = chain;
        }
        end_block();
        s->body = parse_body({"endset"});
        consume_end_tag("endset");
        return s;
    }

    void parse_params(std::vector<Param>& out) {
        expect_op("(");
        while (!is_op(")")) {
            Param p;
            p.name = expect_name();
            if (skip_op("=")) p.def = parse_expression();
            out.push_back(p);
            if (!skip_op(",")) break;
        }
        expect_op(")");
    }

    StmtP parse_macro(int line) {
        auto s = mks(Stmt::T::Macro, line);
        s->text = expect_name();
        if (is_op("(")) parse_params(s->params);
        end_block();
        s->body = parse_body({"endmacro"});
        consume_end_tag("endmacro");
        return s;
    }

    StmtP parse_call(int line) {
        auto s = mks(Stmt::T::CallBlock, line);
        if (is_op("(")) parse_params(s->params);
        s->expr = parse_expression();
        if (s->expr->t != Expr::T::Call) fail("expected call expression in call block", line);
        end_block();
        s->body = parse_body({"endcall"});
        consume_end_tag("endcall");
        return s;
    }

    StmtP parse_filter_block(int line) {
        auto s = mks(Stmt::T::FilterBlock, line);
        ExprP chain;
        do {
            auto f = mk(Expr::T::Filter, cur().line);
            f->name = expect_name();
            f->a = chain;
            if (is_op("(")) parse_call_args(f->args, f->kwargs);
            chain = f;
        } while (skip_op("|"));
        s->expr = chain;
        end_block();
        s->body = parse_body({"endfilter"});
        consume_end_tag("endfilter");
        return s;
    }

    // -------- expressions
    ExprP parse_expression() { return parse_cond(); }

    ExprP parse_cond() {
        ExprP e1 = parse_or();
        while (is_name("if")) {
            int line = cur().line;
            ++p_;
            ExprP e2 = parse_or();
            ExprP e3;
            if (skip_name("else")) e3 = parse_cond();
            auto c = mk(Expr::T::Cond, line);
            c->a = e1; c->b = e2; c->c = e3;
            e1 = c;
        }
        return e1;
    }
    ExprP parse_or() {
        ExprP l = parse_and();
        while (is_name("or")) {
            int line = cur().line;
            ++p_;
            auto e = mk(Expr::T::Or, line);
            e->a = l; e->b = parse_and();
            l = e;
        }
        return l;
    }
    ExprP parse_and() {
        ExprP l = parse_not();
        while (is_name("and")) {
            int line = cur().line;
            ++p_;
            auto e = mk(Expr::T::And, line);
            e->a = l; e->b = parse_not();
            l = e;
        }
        return l;
    }
    ExprP parse_not() {
        if (is_name("not")) {
            int line = cur().line;
            ++p_;
            auto e = mk(Expr::T::Not, line);
            e->a = parse_not();
            return e;
        }
        return parse_compare();
    }
    ExprP parse_compare() {
        ExprP l = parse_math1();
        std::vector<std::pair<Op, ExprP>> chain;
        int line = cur().line;
        for (;;) {
            Op op;
            if (is_op("==")) op = Op::Eq;
            else if (is_op("!=")) op = Op::Ne;
            else if (is_op("<")) op = Op::Lt;
            else if (is_op("<=")) op = Op::Le;
            else if (is_op(">")) op = Op::Gt;
            else if (is_op(">=")) op = Op::Ge;
            else if (is_name("in")) op = Op::In;
            else if (is_name("not") && is_name("in", 1)) { ++p_; op = Op::NotIn; }
            else break;
            ++p_;
            chain.emplace_back(op, parse_math1());
        }
        if (chain.empty()) return l;
        auto e = mk(Expr::T::Compare, line);
        e->a = l;
        e->cmp = std::move(chain);
        return e;
    }
    ExprP bin(Op op, ExprP a, ExprP b, int line) {
        auto e = mk(Expr::T::Bin, line);
        e->op = op; e->a = a; e->b = b;
        return e;
    }
    ExprP parse_math1() {
        ExprP l = parse_concat();
        while (is_op("+") || is_op("-")) {
            int line = cur().line;
            Op op = cur().s == "+" ? Op::Add : Op::Sub;
            ++p_;
            l = bin(op, l, parse_concat(), line);
        }
        return l;
    }
    ExprP parse_concat() {
        ExprP l = parse_math2();
        while (is_op("~")) {
            int line = cur().line;
            ++p_;
            l = bin(Op::Concat, l, parse_math2(), line);
        }
        return l;
    }
    ExprP parse_math2() {
        ExprP l = parse_pow();
        while (is_op("*") || is_op("/") || is_op("//") || is_op("%")) {
            int line = cur().line;
            const std::string& s = cur().s;
            Op op = s == "*" ? Op::Mul : s == "/" ? Op::Div : s == "//" ? Op::FloorDiv : Op::Mod;
            ++p_;
            l = bin(op, l, parse_pow(), line);
        }
        return l;
    }
    ExprP parse_pow() {
        ExprP l = parse_unary(true);
        while (is_op("**")) {
            int line = cur().line;
            ++p_;
            l = bin(Op::Pow, l, parse_unary(true), line);
        }
        return l;
    }
    ExprP parse_unary(bool with_filter) {
        int line = cur().line;
        ExprP n;
        if (is_op("-")) { ++p_; n = mk(Expr::T::Neg, line); n->a = parse_unary(false); }
        else if (is_op("+")) { ++p_; n = mk(Expr::T::Pos, line); n->a = parse_unary(false); }
        else n = parse_primary();
        n = parse_postfix(n);
        if (with_filter) n = parse_filter_expr(n);
        return n;
    }

    ExprP parse_primary() {
        const Token& t = cur();
        int line = t.line;
        switch (t.t) {
            case TT::Name: {
                std::string n = t.s;
                ++p_;
                if (n == "true" || n == "True") { auto e = mk(Expr::T::Const, line); e->cval = Value(true); return e; }
                if (n == "false" || n == "False") { auto e = mk(Expr::T::Const, line); e->cval = Value(false); return e; }
                if (n == "none" || n == "None") { auto e = mk(Expr::T::Const, line); e->cval = Value::none(); return e; }
                auto e = mk(Expr::T::Name, line);
                e->name = n;
                return e;
            }
            case TT::Str: {
                std::string s = t.s;
                ++p_;
                while (cur().t == TT::Str) { s += cur().s; ++p_; }
                auto e = mk(Expr::T::Const, line);
                e->cval = Value(s);
                return e;
            }
            case TT::Int: { auto e = mk(Expr::T::Const, line); e->cval = Value((long long)t.i); ++p_; return e; }
            case TT::Float: { auto e = mk(Expr::T::Const, line); e->cval = Value(t.d); ++p_; return e; }
            case TT::Op:
                if (t.s == "(") {
                    ++p_;
                    if (skip_op(")")) return mk(Expr::T::Tuple, line);
                    ExprP first = parse_expression();
                    if (is_op(",")) {
                        auto tp = mk(Expr::T::Tuple, line);
                        tp->args.push_back(first);
                        while (skip_op(",")) { if (is_op(")")) break; tp->args.push_back(parse_expression()); }
                        expect_op(")");
                        return tp;
                    }
                    expect_op(")");
                    return first;
                }
                if (t.s == "[") {
                    ++p_;
                    auto l = mk(Expr::T::List, line);
                    while (!is_op("]")) {
                        l->args.push_back(parse_expression());
                        if (!skip_op(",")) break;
                    }
                    expect_op("]");
                    return l;
                }
                if (t.s == "{") {
                    ++p_;
                    auto d = mk(Expr::T::Dict, line);
                    while (!is_op("}")) {
                        d->args.push_back(parse_expression());
                        expect_op(":");
                        d->args.push_back(parse_expression());
                        if (!skip_op(",")) break;
                    }
                    expect_op("}");
                    return d;
                }
                break;
            default: break;
        }
        fail("unexpected '" + describe() + "'", line);
    }

    void parse_call_args(std::vector<ExprP>& args, std::vector<std::pair<std::string, ExprP>>& kw) {
        expect_op("(");
        while (!is_op(")")) {
            if (cur().t == TT::Name && is_op("=", 1)) {
                std::string k = next().s;
                ++p_;
                kw.emplace_back(k, parse_expression());
            } else {
                if (is_op("*") || is_op("**")) fail("argument unpacking is not supported", cur().line);
                if (!kw.empty()) fail("positional argument follows keyword argument", cur().line);
                args.push_back(parse_expression());
            }
            if (!skip_op(",")) break;
        }
        expect_op(")");
    }

    ExprP parse_postfix(ExprP n) {
        for (;;) {
            int line = cur().line;
            if (is_op(".")) {
                ++p_;
                auto a = mk(Expr::T::Attr, line);
                a->a = n;
                if (cur().t == TT::Name) a->name = next().s;
                else if (cur().t == TT::Int) {  // x.0
                    auto it = mk(Expr::T::Item, line);
                    it->a = n;
                    auto c = mk(Expr::T::Const, line);
                    c->cval = Value((long long)next().i);
                    it->b = c;
                    n = it;
                    continue;
                } else fail("expected name after '.'", line);
                n = a;
            } else if (is_op("[")) {
                ++p_;
                ExprP idx = parse_subscript();
                expect_op("]");
                auto it = mk(idx->t == Expr::T::Slice ? Expr::T::Item : Expr::T::Item, line);
                it->a = n;
                it->b = idx;
                n = it;
            } else if (is_op("(")) {
                auto c = mk(Expr::T::Call, line);
                c->a = n;
                parse_call_args(c->args, c->kwargs);
                n = c;
            } else break;
        }
        return n;
    }

    ExprP parse_subscript() {
        int line = cur().line;
        ExprP start, stop, step;
        if (!is_op(":")) start = parse_expression();
        if (!is_op(":")) return start;
        auto sl = mk(Expr::T::Slice, line);
        ++p_;
        if (!is_op("]") && !is_op(":")) stop = parse_expression();
        if (skip_op(":")) { if (!is_op("]")) step = parse_expression(); }
        sl->a = start; sl->b = stop; sl->c = step;
        return sl;
    }

    std::string dotted_name() {
        std::string n = expect_name();
        while (is_op(".") && at(1).t == TT::Name) { ++p_; n += "." + next().s; }
        return n;
    }

    ExprP parse_filter_expr(ExprP n) {
        for (;;) {
            int line = cur().line;
            if (is_op("|")) {
                ++p_;
                auto f = mk(Expr::T::Filter, line);
                f->a = n;
                f->name = dotted_name();
                if (is_op("(")) parse_call_args(f->args, f->kwargs);
                n = f;
            } else if (is_name("is")) {
                ++p_;
                auto t = mk(Expr::T::Test, line);
                t->a = n;
                if (skip_name("not")) t->negate = true;
                t->name = dotted_name();
                if (is_op("(")) parse_call_args(t->args, t->kwargs);
                else if ((cur().t == TT::Name || cur().t == TT::Str || cur().t == TT::Int || cur().t == TT::Float || is_op("(") || is_op("[") || is_op("{")) &&
                         !(cur().t == TT::Name && (cur().s == "else" || cur().s == "or" || cur().s == "and" || cur().s == "if" || cur().s == "is" || cur().s == "in" || cur().s == "not"))) {
                    ExprP arg = parse_primary();
                    arg = parse_postfix(arg);
                    t->args.push_back(arg);
                }
                n = t;
            } else break;
        }
        return n;
    }
};

}  // namespace

std::shared_ptr<TemplateAst> parse_template(std::string_view src, const std::string& name, const ParseOptions& opt) {
    Lexer lx(src, opt);
    Parser ps(lx.run());
    auto ast = std::make_shared<TemplateAst>();
    ast->name = name;
    ast->body = ps.parse_all();
    return ast;
}

}  // namespace qv::jinja
