// Legacy compatibility Aurora bridge.
//
// The main live runtime path now uses the Python-native NDITracker backend via
// scikit-surgerynditracker. This executable is retained only for side-by-side
// migration debugging or lab environments that still need the CombinedApi
// bridge path.

#include <CombinedApi.h>

#include <chrono>
#include <csignal>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

struct Options {
    std::string aurora_port = "/dev/ttyUSB0";
    std::string socket_path = "/tmp/tracker_bridge.sock";
    int poll_ms = 20;
};

volatile std::sig_atomic_t g_should_exit = 0;

void handle_signal(int) {
    g_should_exit = 1;
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    out << "\\u" << std::hex << std::uppercase << static_cast<int>(ch);
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

std::string iso8601_utc_now() {
    auto now = std::chrono::system_clock::now();
    auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - seconds).count();

    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &tt);
#else
    gmtime_r(&tt, &tm_utc);
#endif

    char stamp[64];
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", &tm_utc);

    std::ostringstream out;
    out << stamp << ".";
    if (ms < 100) {
        out << "0";
    }
    if (ms < 10) {
        out << "0";
    }
    out << ms << "Z";
    return out.str();
}

class UnixSocketPublisher {
   public:
    explicit UnixSocketPublisher(std::string socket_path)
        : socket_path_(std::move(socket_path)) {}

    ~UnixSocketPublisher() {
        close_all();
    }

    void start() {
        close_all();

        server_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (server_fd_ < 0) {
            throw std::runtime_error("Failed to create Unix socket");
        }

        ::unlink(socket_path_.c_str());

        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        if (socket_path_.size() >= sizeof(addr.sun_path)) {
            throw std::runtime_error("Socket path too long: " + socket_path_);
        }
        std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

        if (::bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            throw std::runtime_error("Failed to bind Unix socket at " + socket_path_);
        }
        if (::listen(server_fd_, 1) < 0) {
            throw std::runtime_error("Failed to listen on Unix socket");
        }

        int flags = ::fcntl(server_fd_, F_GETFL, 0);
        if (flags >= 0) {
            ::fcntl(server_fd_, F_SETFL, flags | O_NONBLOCK);
        }
    }

    void poll_accept() {
        if (server_fd_ < 0 || client_fd_ >= 0) {
            return;
        }
        int fd = ::accept(server_fd_, nullptr, nullptr);
        if (fd < 0) {
            return;
        }
        int flags = ::fcntl(fd, F_GETFL, 0);
        if (flags >= 0) {
            ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        }
        client_fd_ = fd;
    }

    void send_line(const std::string& line) {
        poll_accept();
        if (client_fd_ < 0) {
            return;
        }

        std::string payload = line;
        payload.push_back('\n');
        ssize_t sent = ::send(client_fd_, payload.data(), payload.size(), MSG_NOSIGNAL);
        if (sent < 0) {
            ::close(client_fd_);
            client_fd_ = -1;
        }
    }

   private:
    void close_all() {
        if (client_fd_ >= 0) {
            ::close(client_fd_);
            client_fd_ = -1;
        }
        if (server_fd_ >= 0) {
            ::close(server_fd_);
            server_fd_ = -1;
        }
        if (!socket_path_.empty()) {
            ::unlink(socket_path_.c_str());
        }
    }

    std::string socket_path_;
    int server_fd_ = -1;
    int client_fd_ = -1;
};

void emit_status(UnixSocketPublisher& publisher,
                 const std::string& level,
                 const std::string& state,
                 const std::string& message,
                 const std::string& details_json = "{}") {
    std::ostringstream out;
    out << "{"
        << "\"type\":\"status\"," 
        << "\"timestamp\":\"" << json_escape(iso8601_utc_now()) << "\","
        << "\"level\":\"" << json_escape(level) << "\","
        << "\"state\":\"" << json_escape(state) << "\","
        << "\"message\":\"" << json_escape(message) << "\","
        << "\"details\":" << details_json
        << "}";
    publisher.send_line(out.str());
}

void emit_transform(UnixSocketPublisher& publisher,
                    const std::string& tool_id,
                    uint32_t frame_number,
                    bool valid,
                    const std::string& status,
                    double qw,
                    double qx,
                    double qy,
                    double qz,
                    double tx,
                    double ty,
                    double tz,
                    double quality_error) {
    std::ostringstream out;
    out << "{"
        << "\"type\":\"transform\"," 
        << "\"timestamp\":\"" << json_escape(iso8601_utc_now()) << "\","
        << "\"frame_number\":" << frame_number << ","
        << "\"tool_id\":\"" << json_escape(tool_id) << "\","
        << "\"valid\":" << (valid ? "true" : "false") << ","
        << "\"status\":\"" << json_escape(status) << "\","
        << "\"quaternion\":[" << qw << "," << qx << "," << qy << "," << qz << "],"
        << "\"translation_mm\":[" << tx << "," << ty << "," << tz << "],"
        << "\"quality\":" << quality_error
        << "}";
    publisher.send_line(out.str());
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--aurora-port" && i + 1 < argc) {
            opts.aurora_port = argv[++i];
            continue;
        }
        if (arg == "--socket-path" && i + 1 < argc) {
            opts.socket_path = argv[++i];
            continue;
        }
        if (arg == "--poll-ms" && i + 1 < argc) {
            opts.poll_ms = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: tracker_bridge [--aurora-port /dev/ttyUSB0]"
                      << " [--socket-path /tmp/tracker_bridge.sock]"
                      << " [--poll-ms 20]" << std::endl;
            std::exit(0);
        }
        throw std::runtime_error("Unknown argument: " + arg);
    }
    if (opts.poll_ms < 1) {
        opts.poll_ms = 1;
    }
    return opts;
}

void initialize_and_enable_tools(CombinedApi& capi,
                                 std::vector<ToolData>& enabled_tools,
                                 std::unordered_map<uint16_t, std::string>& handle_to_id,
                                 UnixSocketPublisher& publisher) {
    auto not_init = capi.portHandleSearchRequest(PortHandleSearchRequestOption::NotInit);
    emit_status(
        publisher,
        "info",
        "tools_found",
        "Discovered tool handles requiring initialization",
        "{\"count\":" + std::to_string(not_init.size()) + "}"
    );

    for (const auto& ph : not_init) {
        const std::string handle = ph.getPortHandle();
        int init_rc = capi.portHandleInitialize(handle);
        if (init_rc < 0) {
            emit_status(
                publisher,
                "warning",
                "warning",
                "Tool initialization failed",
                "{\"tool_id\":\"" + json_escape(handle) + "\",\"error\":\"" + json_escape(capi.errorToString(init_rc)) + "\"}"
            );
            continue;
        }

        int enable_rc = capi.portHandleEnable(handle);
        if (enable_rc < 0) {
            emit_status(
                publisher,
                "warning",
                "warning",
                "Tool enable failed",
                "{\"tool_id\":\"" + json_escape(handle) + "\",\"error\":\"" + json_escape(capi.errorToString(enable_rc)) + "\"}"
            );
            continue;
        }

        emit_status(
            publisher,
            "info",
            "tool_enabled",
            "Tool enabled",
            "{\"tool_id\":\"" + json_escape(handle) + "\"}"
        );
    }

    auto enabled = capi.portHandleSearchRequest(PortHandleSearchRequestOption::Enabled);
    enabled_tools.clear();
    handle_to_id.clear();
    for (const auto& ph : enabled) {
        ToolData tool;
        tool.transform.toolHandle = static_cast<uint16_t>(capi.stringToInt(ph.getPortHandle()));
        tool.toolInfo = ph.getPortHandle();
        enabled_tools.push_back(tool);
        handle_to_id[tool.transform.toolHandle] = ph.getPortHandle();
    }

    emit_status(
        publisher,
        "info",
        "tools_ready",
        "Enabled tools ready for tracking",
        "{\"count\":" + std::to_string(enabled_tools.size()) + "}"
    );
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_args(argc, argv);

        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        UnixSocketPublisher publisher(opts.socket_path);
        publisher.start();

        emit_status(
            publisher,
            "info",
            "starting",
            "tracker_bridge starting",
            "{\"aurora_port\":\"" + json_escape(opts.aurora_port) + "\",\"socket_path\":\"" + json_escape(opts.socket_path) + "\"}"
        );

        CombinedApi capi;

        emit_status(
            publisher,
            "info",
            "connecting",
            "Connecting to Aurora",
            "{\"aurora_port\":\"" + json_escape(opts.aurora_port) + "\"}"
        );
        if (capi.connect(opts.aurora_port) != 0) {
            emit_status(publisher, "error", "error", "Failed to connect to Aurora", "{}");
            return 2;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));

        int init_rc = capi.initialize();
        if (init_rc < 0) {
            emit_status(
                publisher,
                "error",
                "error",
                "CombinedApi initialize failed",
                "{\"error\":\"" + json_escape(capi.errorToString(init_rc)) + "\"}"
            );
            return 2;
        }

        emit_status(publisher, "info", "initialized", "CombinedApi initialized", "{}");

        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::vector<ToolData> enabled_tools;
        std::unordered_map<uint16_t, std::string> handle_to_id;
        initialize_and_enable_tools(capi, enabled_tools, handle_to_id, publisher);

        int tracking_rc = capi.startTracking();
        if (tracking_rc < 0) {
            emit_status(
                publisher,
                "error",
                "error",
                "Failed to start tracking",
                "{\"error\":\"" + json_escape(capi.errorToString(tracking_rc)) + "\"}"
            );
            return 2;
        }
        emit_status(publisher, "info", "tracking_started", "Tracking started", "{}");

        while (!g_should_exit) {
            publisher.poll_accept();

            std::vector<ToolData> data = capi.getTrackingDataBX(
                TrackingReplyOption::TransformData | TrackingReplyOption::AllTransforms
            );

            for (auto& tool : enabled_tools) {
                for (const auto& sample : data) {
                    if (sample.transform.toolHandle == tool.transform.toolHandle) {
                        tool = sample;
                        break;
                    }
                }

                const auto it = handle_to_id.find(tool.transform.toolHandle);
                const std::string tool_id = (it != handle_to_id.end()) ? it->second : std::to_string(tool.transform.toolHandle);

                const bool valid = !tool.transform.isMissing();
                const std::string status = valid ? "tracked" : "missing";
                emit_transform(
                    publisher,
                    tool_id,
                    static_cast<uint32_t>(tool.frameNumber),
                    valid,
                    status,
                    static_cast<double>(tool.transform.q0),
                    static_cast<double>(tool.transform.qx),
                    static_cast<double>(tool.transform.qy),
                    static_cast<double>(tool.transform.qz),
                    static_cast<double>(tool.transform.tx),
                    static_cast<double>(tool.transform.ty),
                    static_cast<double>(tool.transform.tz),
                    static_cast<double>(tool.transform.error)
                );
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(opts.poll_ms));
        }

        capi.stopTracking();
        emit_status(publisher, "info", "tracking_stopped", "Tracking stopped", "{}");
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "tracker_bridge fatal error: " << exc.what() << std::endl;
        return 1;
    }
}
