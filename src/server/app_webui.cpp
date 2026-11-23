// Web UI routes, mirroring llama-server's server-http.cpp "Llama Chat setup": embedded gzip assets with ETag / Cache-Control,
// 415 when the client does not accept gzip, self-removing service worker when the UI is off or replaced by --path.
#include <sys/stat.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "server/app.h"
#include "server/webui.h"

namespace qv::server {

namespace {
const std::unordered_map<std::string, const WebuiAsset*>& asset_map() {
    static const auto m = [] {
        std::unordered_map<std::string, const WebuiAsset*> r;
        size_t n = 0;
        const WebuiAsset* a = webui_assets(&n);
        for (size_t i = 0; i < n; ++i) r[a[i].name] = &a[i];
        return r;
    }();
    return m;
}

const char* kSwRemove = R"(
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil((async () => {
    await self.registration.unregister();
    for (const key of await caches.keys()) await caches.delete(key);
    for (const c of await self.clients.matchAll({ type: 'window' })) c.navigate(c.url);
})()));
)";

std::string mime_by_ext(const std::string& p) {
    auto ends = [&](const char* e) { size_t n = std::strlen(e); return p.size() >= n && p.compare(p.size() - n, n, e) == 0; };
    if (ends(".html") || ends(".htm")) return "text/html; charset=utf-8";
    if (ends(".js") || ends(".mjs")) return "text/javascript; charset=utf-8";
    if (ends(".css")) return "text/css; charset=utf-8";
    if (ends(".json") || ends(".map")) return "application/json; charset=utf-8";
    if (ends(".webmanifest")) return "application/manifest+json";
    if (ends(".svg")) return "image/svg+xml";
    if (ends(".png")) return "image/png";
    if (ends(".jpg") || ends(".jpeg")) return "image/jpeg";
    if (ends(".ico")) return "image/x-icon";
    if (ends(".woff2")) return "font/woff2";
    if (ends(".woff")) return "font/woff";
    if (ends(".txt")) return "text/plain; charset=utf-8";
    return "application/octet-stream";
}
}  // namespace

bool App::is_frontend_path(const std::string& path) {
    return path == "/" || (path.size() > 1 && asset_map().count(path.substr(1)) != 0);
}

// Returns true when the response was written.
bool App::serve_webui(const http::Request& req, http::Conn& c, const std::string& path) {
    const bool sw = path == "/sw.js" || path == "/service-worker.js";
    auto sw_remove = [&] { c.respond(200, "application/javascript", kSwRemove, {{"Cache-Control", "no-cache"}}); return true; };
    if (!o.webui) return sw ? sw_remove() : false;

    if (!o.path.empty()) {  // --path DIR: plain files from disk (cpp-httplib mount point semantics)
        std::string p = path == "/" ? "/index.html" : path;
        if (p.find("..") == std::string::npos) {
            std::string full = o.path + p;
            struct stat st;
            if (::stat(full.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) { full += "/index.html"; p += "/index.html"; }
            std::ifstream f(full, std::ios::binary);
            if (f) {
                std::stringstream ss;
                ss << f.rdbuf();
                c.respond(200, mime_by_ext(p), ss.str());
                return true;
            }
        }
        return sw ? sw_remove() : false;
    }

    const std::string name = path == "/" ? "index.html" : path.substr(1);
    auto it = asset_map().find(name);
    if (it == asset_map().end()) return false;
    const WebuiAsset& a = *it->second;
    if (req.header("accept-encoding").find("gzip") == std::string::npos) {
        c.respond(415, "text/plain", "Error: gzip is not supported by this browser");
        return true;
    }
    static const std::unordered_set<std::string> no_cache = {"service-worker.js", "sw.js", "manifest.webmanifest", "_app/version.json", "build.json"};
    const bool is_index = name == "index.html";
    http::Headers h{{"Content-Encoding", "gzip"}};
    if (no_cache.count(name)) {
        h.push_back({"Cache-Control", "no-cache"});
    } else {
        h.push_back({"ETag", a.etag});
        const std::string inm = req.header("if-none-match");
        if (!inm.empty() && (inm == a.etag || inm == std::string("W/") + a.etag)) {
            c.respond(304, "", "", h);
            return true;
        }
        if (is_index) {
            h.push_back({"Cross-Origin-Embedder-Policy", "require-corp"});
            h.push_back({"Cross-Origin-Opener-Policy", "same-origin"});
        }
        h.push_back({"Cache-Control", is_index ? "no-cache" : "public, max-age=31536000, immutable"});
    }
    c.respond(200, a.type, std::string_view(reinterpret_cast<const char*>(a.data), a.size), h);
    return true;
}

}  // namespace qv::server
