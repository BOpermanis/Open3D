// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "open3d/utility/FileSystem.h"

namespace open3d {

namespace core {
class Tensor;
}

namespace visualization {
namespace gui {
struct MouseEvent;
class Window;
}  // namespace gui

namespace webrtc_server {

class WebRTCServer {
public:
    static WebRTCServer& GetInstance();

    void StartWebRTCServer();

    /// Client -> server message.
    void OnDataChannelMessage(const std::string& message);

    /// Set MouseEvent callback function. If a client -> server message is of
    /// MouseEvent type, the callback function will be triggered. The client
    /// message shall also contain the corresponding window_uid.
    void SetMouseEventCallback(
            std::function<void(const std::string&, const gui::MouseEvent&)> f);

    /// Set redraw callback function. Server can force a redraw. Then redraw
    /// then triggers OnFrame(), where a server -> client frame will be sent.
    void SetRedrawCallback(std::function<void(const std::string&)> f);

    /// Server -> client frame.
    void OnFrame(const std::string& window_uid,
                 const std::shared_ptr<core::Tensor>& im);

    /// Send initial frames. This flushes the WebRTC video stream. After the
    /// initial frames, new frames will only be sent at triggered events.
    void SendInitFrames(const std::string& window_uid);

    /// List available windows.
    std::vector<std::string> GetWindowUIDs() const;

    /// Call PeerConnectionManager's web request API.
    /// This function is called in JavaScript via Python binding to mimic the
    /// behavior of sending HTTP request via fetch() in JavaScript.
    ///
    /// With fetch:
    /// data = {method: "POST", body: JSON.stringify(candidate)};
    /// fetch(this.srvurl + "/api/addIceCandidate?peerid=" + peerid, data);
    ///
    /// Now with CallHttpAPI:
    /// open3d.visualization.webrtc_server("/api/addIceCandidate",
    ///                                    "?peerid=" + peerid,
    ///                                    data["body"]);
    ///
    /// \param entry_point URL part before '?'.
    /// \param query_string URL part after '?', including '?'. If '?' is not the
    /// first character or if the stirng is empty, the query_string is ignored.
    /// \param data JSON-encoded string.
    std::string CallHttpAPI(const std::string& entry_point,
                            const std::string& query_string = "",
                            const std::string& data = "") const;

    /// Sets WebRTCWindowSystem as the default window system in Application.
    /// This enables a global WebRTC server and each gui::Window will be
    /// rendered to a WebRTC video stream.
    void EnableWebRTC();

    /// HTTP handshake server is enabled by default. Call DisableHttpServer() to
    /// disable the HTTP server. This must be called before WebRTCServer::Run(),
    /// i.e. before WebRTCWindowSystem::StartWebRTCServer() or
    /// Application::AddWindow().
    void DisableHttpHandshake();

    /// Close all WebRTC connections that correspond to a Window.
    void CloseWindowConnections(const std::string& window_uid);

    /// Cleans up WebRTC peer connection and thread.
    ~WebRTCServer();

private:
    WebRTCServer();
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
