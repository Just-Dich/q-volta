// Minimal HTTP/1.1 server on POSIX sockets: accept loop + one thread per connection (capped), keep-alive, Content-Length bodies,
// chunked responses for streaming, client-disconnect detection. Response framing mimics cpp-httplib as used by llama-server
// (Keep-Alive header, `Connection: close` on errors, HEAD without body).
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace qv::http {

using Headers = std::vector<std::pair<std::string, std::string>>;

struct Request {
    std::string method, path, query, body;
    std::map<std::string, std::string> headers;  // lower-case names
    std::string header(const std::string& k) const {
        auto it = headers.find(k);
        return it == headers.end() ? std::string() : it->second;
    }
    bool has_header(const std::string& k) const { return headers.count(k) != 0; }
    std::string param(const std::string& name) const;  // URL query parameter (percent-decoded), "" if absent
};

class Conn {
   public:
    explicit Conn(int fd) : fd_(fd) {}
    bool send_all(std::string_view s);  // false on error / peer gone
    bool peer_closed();                 // non-blocking check for FIN/RST
    // Whole response with Content-Length (status >= 400 closes the connection, like llama-server).
    bool respond(int status, const std::string& content_type, std::string_view body, const Headers& extra = {});
    // Streaming: headers with Transfer-Encoding: chunked, then write_chunk()*, then end_chunks().
    bool begin_stream(int status, const std::string& content_type, const Headers& extra = {});
    bool write_chunk(std::string_view data);
    bool end_chunks();
    bool close_after = false;
    bool broken = false;
    bool head_only = false;
    bool headers_sent = false, streaming = false;
    Headers base_headers;  // sent with every response (Server, Access-Control-Allow-Origin, ...)
    int fd() const { return fd_; }

   private:
    int fd_;
    std::string head(int status, const std::string& ct, const Headers& extra, const std::string& framing);
};

using Handler = std::function<void(const Request&, Conn&)>;

class Server {
   public:
    Server(std::string host, int port, int max_conns, Handler h) : host_(std::move(host)), port_(port), max_conns_(max_conns > 0 ? max_conns : 256), h_(std::move(h)) {}
    ~Server();
    void set_timeout(int seconds) { timeout_s_ = seconds; }
    void set_base_headers(Headers h) { base_ = std::move(h); }
    bool listen_on(std::string& err);  // bind + listen
    void run();                        // blocks until stop()
    void stop();
    int port() const { return port_; }

   private:
    std::string host_;
    int port_, max_conns_, lfd_ = -1, timeout_s_ = 3600;
    Handler h_;
    Headers base_;
    std::atomic<bool> stop_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    int active_ = 0;
    void serve(int fd);
};

const char* status_text(int s);

}  // namespace qv::http
