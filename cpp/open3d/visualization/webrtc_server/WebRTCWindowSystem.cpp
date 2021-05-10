// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/visualization/webrtc_server/WebRTCWindowSystem.h"

#include <chrono>
#include <sstream>
#include <thread>
#include <unordered_map>

#include "open3d/core/Tensor.h"
#include "open3d/io/ImageIO.h"
#include "open3d/utility/Console.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/gui/Events.h"
#include "open3d/visualization/utility/Draw.h"
#include "open3d/visualization/webrtc_server/WebRTCServer.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

struct WebRTCWindowSystem::Impl {
    std::thread webrtc_thread_;
    bool sever_started_ = false;
    std::unordered_map<WebRTCWindowSystem::OSWindow, std::string>
            os_window_to_uid_;
    std::string GenerateUID() {
        static std::atomic<size_t> count{0};
        return "window_" + std::to_string(count++);
    }
};

std::shared_ptr<WebRTCWindowSystem> WebRTCWindowSystem::GetInstance() {
    static std::shared_ptr<WebRTCWindowSystem> instance(new WebRTCWindowSystem);
    return instance;
}

WebRTCWindowSystem::WebRTCWindowSystem()
    : BitmapWindowSystem(
#if !defined(__APPLE__) && !defined(_WIN32) && !defined(_WIN64)
              BitmapWindowSystem::Rendering::HEADLESS
#else
              BitmapWindowSystem::Rendering::NORMAL
#endif
              ),
      impl_(new WebRTCWindowSystem::Impl()) {

    // Server->client send frame.
    auto draw_callback = [this](const gui::Window *window,
                                std::shared_ptr<core::Tensor> im) -> void {
        WebRTCServer::GetInstance().OnFrame(
                WebRTCWindowSystem::GetInstance()->GetWindowUID(
                        window->GetOSWindow()),
                im);
    };
    SetOnWindowDraw(draw_callback);

    // Client -> server message can trigger a mouse event and
    // mouse_event_callback will be called.
    auto mouse_event_callback = [this](const std::string &window_uid,
                                       const gui::MouseEvent &me) -> void {
        this->PostMouseEvent(
                WebRTCWindowSystem::GetInstance()->GetOSWindowByUID(window_uid),
                me);
    };
    this->SetMouseEventCallback(mouse_event_callback);

    // redraw_callback is called when the server wants to send a frame to
    // the client without other triggering events.
    auto redraw_callback = [this](const std::string &window_uid) -> void {
        this->PostRedrawEvent(
                WebRTCWindowSystem::GetInstance()->GetOSWindowByUID(
                        window_uid));
    };
    this->SetRedrawCallback(redraw_callback);
}

WebRTCWindowSystem::~WebRTCWindowSystem() {}

WebRTCWindowSystem::OSWindow WebRTCWindowSystem::CreateOSWindow(
        gui::Window *o3d_window,
        int width,
        int height,
        const char *title,
        int flags) {
    // No-op if the server is already running.
    StartWebRTCServer();
    WebRTCWindowSystem::OSWindow os_window = BitmapWindowSystem::CreateOSWindow(
            o3d_window, width, height, title, flags);
    std::string window_uid = impl_->GenerateUID();
    impl_->os_window_to_uid_.insert({os_window, window_uid});
    utility::LogInfo("Window {} created.", window_uid);
    return os_window;
}

void WebRTCWindowSystem::DestroyWindow(OSWindow w) {
    std::string window_uid = impl_->os_window_to_uid_.at(w);
    CloseWindowConnections(window_uid);
    impl_->os_window_to_uid_.erase(w);
    BitmapWindowSystem::DestroyWindow(w);
    utility::LogInfo("Window {} destroyed.", window_uid);
}

std::vector<std::string> WebRTCWindowSystem::GetWindowUIDs() const {
    std::vector<std::string> uids;
    for (const auto &it : impl_->os_window_to_uid_) {
        uids.push_back(it.second);
    }
    return uids;
}

std::string WebRTCWindowSystem::GetWindowUID(
        WebRTCWindowSystem::OSWindow w) const {
    if (impl_->os_window_to_uid_.count(w) == 0) {
        return "window_undefined";
    } else {
        return impl_->os_window_to_uid_.at(w);
    }
}

WebRTCWindowSystem::OSWindow WebRTCWindowSystem::GetOSWindowByUID(
        const std::string &uid) const {
    // This can be optimized by adding a bi-directional map, but it may not be
    // worth it since we typically don't have lots of windows.
    for (const auto &it : impl_->os_window_to_uid_) {
        if (it.second == uid) {
            return it.first;
        };
    }
    return nullptr;
}

void WebRTCWindowSystem::SetMouseEventCallback(
        std::function<void(const std::string &, const gui::MouseEvent &)> f) {
    WebRTCServer::GetInstance().SetMouseEventCallback(f);
}

void WebRTCWindowSystem::SetRedrawCallback(
        std::function<void(const std::string &)> f) {
    WebRTCServer::GetInstance().SetRedrawCallback(f);
}

void WebRTCWindowSystem::StartWebRTCServer() {
    if (!impl_->sever_started_) {
        auto start_webrtc_thread = [this]() {
            WebRTCServer::GetInstance().StartWebRTCServer();
        };
        impl_->webrtc_thread_ = std::thread(start_webrtc_thread);
        impl_->sever_started_ = true;
    }
}

void WebRTCWindowSystem::CloseWindowConnections(const std::string &window_uid) {
    WebRTCServer::GetInstance().CloseWindowConnections(window_uid);
}

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
