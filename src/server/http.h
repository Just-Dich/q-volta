// Minimal HTTP/1.1 server on POSIX sockets: accept loop + fixed worker pool, keep-alive, Content-Length bodies,
// chunked responses for streaming, client-disconnect detection.
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace qv::http {

struct Request {
    std::string method, path, query, body;
    std::map<std::string, std::string> headers;  // lower-case names
    std::string header(const std::string& k) const {
        auto it = headers.find(k);
        return it == headers.end() ? std::string() : it->second;
    }
};

class Conn {
   public:
    explicit Conn(int fd) : fd_(fd) {}
    bool send_all(std::string_view s);  // false on error / peer gone
    bool peer_closed();                 // non-blocking check for FIN/RST
    // Whole response with Content-Length.
    bool respond(int status, const std::string& content_type, const std::string& body, const std::vector<std::pair<std::string, std::string>>& extra = {});
    // Streaming: headers with Transfer-Encoding: chunked, then write_chunk()*, then end_chunks().
    bool begin_stream(int status, const std::string& content_type, const std::vector<std::pair<std::string, std::string>>& extra = {});
    bool write_chunk(std::string_view data);
    bool end_chunks();
    bool close_after = false;
    bool broken = false;
    int fd() const { return fd_; }

   private:
    int fd_;
};

using Handler = std::function<void(const Request&, Conn&)>;

class Server {
   public:
    Server(std::string host, int port, int workers, Handler h) : host_(std::move(host)), port_(port), workers_(workers), h_(std::move(h)) {}
    ~Server();
    bool listen_on(std::string& err);  // bind + listen
    void run();                        // blocks until stop()
    void stop();
    int port() const { return port_; }

   private:
    std::string host_;
    int port_, workers_, lfd_ = -1;
    Handler h_;
    std::atomic<bool> stop_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<int> q_;
    std::vector<std::thread> threads_;
    void worker();
    void serve(int fd);
};

const char* status_text(int s);

}  // namespace qv::http
