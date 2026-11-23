#include "server/http.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>

namespace qv::http {

const char* status_text(int s) {
    switch (s) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        default: return "Status";
    }
}

bool Conn::send_all(std::string_view s) {
    if (broken) return false;
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = ::send(fd_, s.data() + off, s.size() - off, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            broken = true;
            return false;
        }
        off += (size_t)n;
    }
    return true;
}

bool Conn::peer_closed() {
    if (broken) return true;
    char c;
    ssize_t n = ::recv(fd_, &c, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n == 0) { broken = true; return true; }
    if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) { broken = true; return true; }
    return false;
}

std::string Conn::head(int status, const std::string& ct, const Headers& extra, const std::string& framing) {
    std::string h = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\n";
    for (auto& [k, v] : base_headers) h += k + ": " + v + "\r\n";
    for (auto& [k, v] : extra) h += k + ": " + v + "\r\n";
    if (!ct.empty()) h += "Content-Type: " + ct + "\r\n";
    h += close_after ? "Connection: close\r\n" : "Keep-Alive: timeout=5, max=100\r\n";
    h += framing;
    return h;
}

bool Conn::respond(int status, const std::string& ct, std::string_view body, const Headers& extra) {
    if (status >= 400) close_after = true;
    headers_sent = true;
    std::string h = head(status, ct, extra, "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n");
    if (!head_only) h += body;
    return send_all(h);
}
bool Conn::begin_stream(int status, const std::string& ct, const Headers& extra) {
    Headers e = extra;
    e.insert(e.begin(), {"X-Accel-Buffering", "no"});
    headers_sent = streaming = true;
    return send_all(head(status, ct, e, "Transfer-Encoding: chunked\r\n\r\n"));
}
bool Conn::write_chunk(std::string_view d) {
    if (d.empty()) return true;
    char hex[24];
    int n = std::snprintf(hex, sizeof hex, "%zx\r\n", d.size());
    std::string s(hex, n);
    s.append(d);
    s += "\r\n";
    return send_all(s);
}
bool Conn::end_chunks() { return send_all("0\r\n\r\n"); }

static std::string url_decode(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() + 0 && std::isxdigit((unsigned char)s[i + 1]) && std::isxdigit((unsigned char)s[i + 2])) { o += (char)std::stoi(s.substr(i + 1, 2), nullptr, 16); i += 2; }
        else if (s[i] == '+') o += ' ';
        else o += s[i];
    }
    return o;
}
std::string Request::param(const std::string& name) const {
    size_t pos = 0;
    while (pos <= query.size()) {
        size_t e = query.find('&', pos);
        if (e == std::string::npos) e = query.size();
        std::string kv = query.substr(pos, e - pos);
        size_t eq = kv.find('=');
        if (url_decode(kv.substr(0, eq)) == name) return eq == std::string::npos ? "" : url_decode(kv.substr(eq + 1));
        pos = e + 1;
    }
    return "";
}

Server::~Server() { stop(); if (lfd_ >= 0) ::close(lfd_); }

bool Server::listen_on(std::string& err) {
    lfd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (lfd_ < 0) { err = std::strerror(errno); return false; }
    int one = 1;
    setsockopt(lfd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port_);
    std::string h = host_ == "localhost" ? "127.0.0.1" : host_;
    if (h.empty() || h == "::" || h == "*") h = "0.0.0.0";
    if (inet_pton(AF_INET, h.c_str(), &a.sin_addr) != 1) { err = "bad --host (IPv4 address expected)"; return false; }
    if (::bind(lfd_, (sockaddr*)&a, sizeof a) < 0 || ::listen(lfd_, 128) < 0) { err = std::strerror(errno); return false; }
    if (port_ == 0) { socklen_t l = sizeof a; getsockname(lfd_, (sockaddr*)&a, &l); port_ = ntohs(a.sin_port); }
    return true;
}

void Server::stop() {
    if (stop_.exchange(true)) return;
    if (lfd_ >= 0) ::shutdown(lfd_, SHUT_RDWR);
    cv_.notify_all();
}

void Server::run() {
    while (!stop_) {
        {
            std::unique_lock<std::mutex> g(mu_);
            cv_.wait(g, [&] { return stop_ || active_ < max_conns_; });
        }
        pollfd p{lfd_, POLLIN, 0};
        int r = ::poll(&p, 1, 250);
        if (r <= 0) continue;
        int fd = ::accept(lfd_, nullptr, nullptr);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        timeval tv{timeout_s_ > 0 ? timeout_s_ : 3600, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        { std::lock_guard<std::mutex> g(mu_); ++active_; }
        std::thread([this, fd] {
            serve(fd);
            ::close(fd);
            { std::lock_guard<std::mutex> g(mu_); --active_; }
            cv_.notify_all();
        }).detach();
    }
    std::unique_lock<std::mutex> g(mu_);
    cv_.wait_for(g, std::chrono::seconds(10), [&] { return active_ == 0; });
}

static constexpr size_t kMaxHeader = 64 * 1024, kMaxBody = 32u * 1024 * 1024;

void Server::serve(int fd) {
    Conn conn(fd);
    conn.base_headers = base_;
    std::string buf;
    char tmp[16384];
    bool first = true;
    while (!stop_) {
        // ---- read headers
        size_t hend;
        while ((hend = buf.find("\r\n\r\n")) == std::string::npos) {
            if (buf.empty() && !first) {  // keep-alive idle timeout (llama-server: 5 s)
                pollfd pf{fd, POLLIN, 0};
                if (::poll(&pf, 1, 5000) <= 0) return;
            }
            if (buf.size() > kMaxHeader) { conn.respond(400, "text/plain", "headers too large"); return; }
            ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
            if (n <= 0) return;  // closed / timeout
            buf.append(tmp, (size_t)n);
        }
        Request req;
        {
            std::string hdr = buf.substr(0, hend);
            size_t le = hdr.find("\r\n");
            std::string line = hdr.substr(0, le);
            size_t s1 = line.find(' '), s2 = line.rfind(' ');
            if (s1 == std::string::npos || s2 == s1) { conn.respond(400, "text/plain", "bad request line"); return; }
            req.method = line.substr(0, s1);
            std::string target = line.substr(s1 + 1, s2 - s1 - 1);
            size_t q = target.find('?');
            req.path = target.substr(0, q);
            if (q != std::string::npos) req.query = target.substr(q + 1);
            size_t pos = le == std::string::npos ? hdr.size() : le + 2;
            while (pos < hdr.size()) {
                size_t e = hdr.find("\r\n", pos);
                if (e == std::string::npos) e = hdr.size();
                std::string l = hdr.substr(pos, e - pos);
                pos = e + 2;
                size_t c = l.find(':');
                if (c == std::string::npos) continue;
                std::string k = l.substr(0, c), v = l.substr(c + 1);
                std::transform(k.begin(), k.end(), k.begin(), [](unsigned char x) { return std::tolower(x); });
                size_t b = v.find_first_not_of(" \t"), t = v.find_last_not_of(" \t");
                req.headers[k] = b == std::string::npos ? "" : v.substr(b, t - b + 1);
            }
        }
        buf.erase(0, hend + 4);
        first = false;
        auto fill = [&]() -> bool { ssize_t n = ::recv(fd, tmp, sizeof tmp, 0); if (n <= 0) return false; buf.append(tmp, (size_t)n); return true; };
        std::string te = req.header("transfer-encoding");
        std::transform(te.begin(), te.end(), te.begin(), [](unsigned char x) { return std::tolower(x); });
        if (te.find("chunked") != std::string::npos) {
            if (req.header("expect") == "100-continue") conn.send_all("HTTP/1.1 100 Continue\r\n\r\n");
            std::string body;
            for (;;) {
                size_t le;
                while ((le = buf.find("\r\n")) == std::string::npos) if (!fill()) return;
                size_t csz;
                try { csz = std::stoull(buf.substr(0, le), nullptr, 16); } catch (...) { conn.respond(400, "text/plain", "bad chunk"); return; }
                buf.erase(0, le + 2);
                if (csz == 0) {  // trailer: consume until the empty line
                    while (buf.find("\r\n") == std::string::npos) if (!fill()) return;
                    size_t t = buf.find("\r\n");
                    while (t != 0) { buf.erase(0, t + 2); while ((t = buf.find("\r\n")) == std::string::npos) if (!fill()) return; }
                    buf.erase(0, 2);
                    break;
                }
                if (body.size() + csz > kMaxBody) { conn.respond(413, "text/plain", "body too large"); return; }
                while (buf.size() < csz + 2) if (!fill()) return;
                body.append(buf, 0, csz);
                buf.erase(0, csz + 2);
            }
            req.body = std::move(body);
        } else {
            size_t clen = 0;
            if (std::string cl = req.header("content-length"); !cl.empty()) {
                try { clen = std::stoull(cl); } catch (...) { conn.respond(400, "text/plain", "bad Content-Length"); return; }
                if (clen > kMaxBody) { conn.respond(413, "text/plain", "body too large"); return; }
            }
            if (clen && buf.size() < clen && req.header("expect") == "100-continue") conn.send_all("HTTP/1.1 100 Continue\r\n\r\n");
            while (buf.size() < clen) if (!fill()) return;
            req.body = buf.substr(0, clen);
            buf.erase(0, clen);
        }
        std::string conn_hdr = req.header("connection");
        std::transform(conn_hdr.begin(), conn_hdr.end(), conn_hdr.begin(), [](unsigned char x) { return std::tolower(x); });
        conn.close_after = conn_hdr == "close";
        conn.head_only = req.method == "HEAD";
        try {
            h_(req, conn);
        } catch (const std::exception& e) {
            if (!conn.broken) conn.respond(500, "application/json; charset=utf-8", std::string("{\"error\":{\"code\":500,\"message\":\"internal error\",\"type\":\"server_error\"}}"));
            conn.close_after = true;
        }
        if (conn.close_after || conn.broken) return;
    }
}

}  // namespace qv::http
