/**
 * http.h — a minimal HTTP/1.1 server, no dependencies (S2.4).
 *
 * Enough to demo the engine: POST a prompt, stream tokens back. Deliberately
 * not a real HTTP library — one thread per connection, no keep-alive, no TLS,
 * no chunked request bodies. The interesting part of this project is the
 * inference engine; this is the socket in front of it.
 *
 * Streaming uses Server-Sent Events (text/event-stream): the response header is
 * sent immediately, then each token is written as
 *
 *     data: {"token":"..."}\n\n
 *
 * and flushed. Browsers and `curl -N` render it as it arrives. SSE rather than
 * chunked JSON because it needs no framing logic on either side.
 */

#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace http {

/// Wire format. OpenAI so existing clients (openai-python, LangChain, any chat
/// UI) work unmodified; Simple for debugging with curl.
enum class Format { OpenAI, Simple };

/// A live connection the engine streams into. Owns the socket.
class Stream {
public:
    Stream(int fd, Format fmt, std::string model)
        : fd_(fd), fmt_(fmt), model_(std::move(model)) {}
    ~Stream() { close(); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    /// Send the SSE response header. Call once, before any token.
    void begin() {
        static const char* hdr =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n";
        write_all(hdr, strlen(hdr));
    }

    /// One token. `text` is raw UTF-8; it gets JSON-escaped.
    void token(const std::string& text) {
        std::string msg;
        if (fmt_ == Format::OpenAI) {
            msg = "data: {\"id\":\"" + id_ + "\",\"object\":\"text_completion\""
                  ",\"model\":\"" + model_ + "\",\"choices\":[{\"index\":0"
                  ",\"text\":\"" + escape(text) + "\",\"finish_reason\":null}]}\n\n";
        } else {
            msg = "data: {\"token\":\"" + escape(text) + "\"}\n\n";
        }
        write_all(msg.data(), msg.size());
    }

    /// `stop` = hit EOS, otherwise the token limit — OpenAI's two finish reasons.
    void done(int n_tokens, double ttft_ms, double tok_s, bool stop) {
        std::string msg;
        if (fmt_ == Format::OpenAI) {
            // Final chunk carries finish_reason and usage, then the sentinel
            // that openai-python waits for.
            char buf[512];
            snprintf(buf, sizeof buf,
                "data: {\"id\":\"%s\",\"object\":\"text_completion\","
                "\"model\":\"%s\",\"choices\":[{\"index\":0,\"text\":\"\","
                "\"finish_reason\":\"%s\"}],"
                "\"usage\":{\"completion_tokens\":%d},"
                "\"timings\":{\"ttft_ms\":%.1f,\"tok_s\":%.1f}}\n\n"
                "data: [DONE]\n\n",
                id_.c_str(), model_.c_str(), stop ? "stop" : "length",
                n_tokens, ttft_ms, tok_s);
            msg = buf;
        } else {
            char buf[256];
            snprintf(buf, sizeof buf,
                "data: {\"done\":true,\"tokens\":%d,\"ttft_ms\":%.1f,"
                "\"tok_s\":%.1f}\n\n", n_tokens, ttft_ms, tok_s);
            msg = buf;
        }
        write_all(msg.data(), msg.size());
        close();
    }

    void set_id(const std::string& id) { id_ = id; }

    /// A client that hung up. The engine checks this to cancel work early.
    bool alive() const { return fd_ >= 0 && !broken_; }

    void close() {
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    }

private:
    void write_all(const char* p, size_t n) {
        while (n > 0) {
            const ssize_t w = ::send(fd_, p, n, MSG_NOSIGNAL);
            if (w <= 0) { broken_ = true; return; }   // client gone
            p += w; n -= (size_t)w;
        }
    }

    static std::string escape(const std::string& s) {
        std::string o;
        o.reserve(s.size() + 8);
        for (unsigned char c : s) {
            switch (c) {
                case '"':  o += "\\\""; break;
                case '\\': o += "\\\\"; break;
                case '\n': o += "\\n";  break;
                case '\r': o += "\\r";  break;
                case '\t': o += "\\t";  break;
                default:
                    if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                    else o += (char)c;
            }
        }
        return o;
    }

    int         fd_ = -1;
    bool        broken_ = false;
    Format      fmt_ = Format::OpenAI;
    std::string model_ = "llm_server";
    std::string id_ = "cmpl-0";
};

struct Request {
    std::string prompt;
    int    max_tokens = 64;
    Format format = Format::OpenAI;
};

/// Handler takes the parsed request and a stream to write into. It is called on
/// the connection's own thread; the engine may take ownership of the stream and
/// finish it later.
using Handler = std::function<void(Request, std::shared_ptr<Stream>)>;

class Server {
public:
    Server() { signal(SIGPIPE, SIG_IGN); }   // MSG_NOSIGNAL covers send(), belt and braces

    void listen_and_serve(int port, Handler h) {
        handler_ = std::move(h);
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) throw std::runtime_error("socket() failed");
        int on = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons((uint16_t)port);
        if (::bind(listen_fd_, (sockaddr*)&addr, sizeof addr) < 0)
            throw std::runtime_error("bind() failed — port in use?");
        if (::listen(listen_fd_, 64) < 0)
            throw std::runtime_error("listen() failed");

        running_ = true;
        accept_thread_ = std::thread([this] { accept_loop(); });
    }

    void stop() {
        running_ = false;
        if (listen_fd_ >= 0) { ::shutdown(listen_fd_, SHUT_RDWR); ::close(listen_fd_); }
        if (accept_thread_.joinable()) accept_thread_.join();
    }

private:
    void accept_loop() {
        while (running_) {
            const int fd = ::accept(listen_fd_, nullptr, nullptr);
            if (fd < 0) { if (running_) continue; else break; }
            int on = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof on);  // stream promptly
            std::thread([this, fd] { serve_one(fd); }).detach();
        }
    }

    void serve_one(int fd) {
        std::string buf;
        char tmp[4096];
        // Read until the header terminator, then the body if Content-Length says so.
        while (buf.find("\r\n\r\n") == std::string::npos) {
            const ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
            if (n <= 0) { ::close(fd); return; }
            buf.append(tmp, (size_t)n);
            if (buf.size() > (1u << 20)) { ::close(fd); return; }
        }
        const size_t hdr_end = buf.find("\r\n\r\n") + 4;
        const size_t want = content_length(buf.substr(0, hdr_end));
        while (buf.size() - hdr_end < want) {
            const ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
            if (n <= 0) break;
            buf.append(tmp, (size_t)n);
        }

        if (buf.rfind("OPTIONS", 0) == 0) {          // CORS preflight
            static const char* r = "HTTP/1.1 204 No Content\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Headers: *\r\n"
                "Access-Control-Allow-Methods: POST, OPTIONS\r\n\r\n";
            ::send(fd, r, strlen(r), MSG_NOSIGNAL);
            ::close(fd);
            return;
        }
        if (buf.rfind("POST", 0) != 0) {
            static const char* r = "HTTP/1.1 405 Method Not Allowed\r\n"
                "Content-Length: 0\r\nConnection: close\r\n\r\n";
            ::send(fd, r, strlen(r), MSG_NOSIGNAL);
            ::close(fd);
            return;
        }

        // Route on path. /v1/completions is the OpenAI-compatible endpoint, so
        // openai-python, LangChain and chat UIs work unmodified; /generate is a
        // simpler shape for curl.
        const bool openai = buf.find("/v1/completions") != std::string::npos;

        const std::string body = buf.substr(hdr_end);
        Request req;
        req.prompt     = json_string(body, "prompt");
        req.max_tokens = json_int(body, "max_tokens", 64);
        req.format     = openai ? Format::OpenAI : Format::Simple;
        if (req.prompt.empty()) {
            static const char* r = "HTTP/1.1 400 Bad Request\r\n"
                "Content-Length: 0\r\nConnection: close\r\n\r\n";
            ::send(fd, r, strlen(r), MSG_NOSIGNAL);
            ::close(fd);
            return;
        }

        // Ownership moves to the engine; it closes the socket when the request
        // completes. This thread ends here.
        auto st = std::make_shared<Stream>(fd, req.format, model_name_);
        handler_(std::move(req), std::move(st));
    }

public:
    void set_model_name(std::string n) { model_name_ = std::move(n); }

private:

    static size_t content_length(const std::string& hdr) {
        const char* key = "content-length:";
        std::string lower;
        lower.reserve(hdr.size());
        for (char c : hdr) lower += (char)tolower((unsigned char)c);
        const size_t p = lower.find(key);
        if (p == std::string::npos) return 0;
        return (size_t)strtoul(hdr.c_str() + p + strlen(key), nullptr, 10);
    }

    // Hand-rolled, same approach as tokenizer.h / model_config.h — no JSON dep.
    static std::string json_string(const std::string& s, const std::string& key) {
        const std::string pat = "\"" + key + "\"";
        size_t p = s.find(pat);
        if (p == std::string::npos) return "";
        p = s.find(':', p + pat.size());
        if (p == std::string::npos) return "";
        p = s.find('"', p);
        if (p == std::string::npos) return "";
        ++p;
        std::string out;
        while (p < s.size() && s[p] != '"') {
            if (s[p] == '\\' && p + 1 < s.size()) {
                ++p;
                switch (s[p]) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    default:  out += s[p];
                }
            } else out += s[p];
            ++p;
        }
        return out;
    }

    static int json_int(const std::string& s, const std::string& key, int def) {
        const std::string pat = "\"" + key + "\"";
        size_t p = s.find(pat);
        if (p == std::string::npos) return def;
        p = s.find(':', p + pat.size());
        if (p == std::string::npos) return def;
        ++p;
        while (p < s.size() && isspace((unsigned char)s[p])) ++p;
        if (p >= s.size() || !isdigit((unsigned char)s[p])) return def;
        return atoi(s.c_str() + p);
    }

    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread accept_thread_;
    Handler handler_;
    std::string model_name_ = "llm_server";
};

}  // namespace http
