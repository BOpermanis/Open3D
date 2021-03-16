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

#include <api/video/i420_buffer.h>
#include <libyuv/convert.h>
#include <libyuv/video_common.h>
#include <media/base/video_broadcaster.h>
#include <media/base/video_common.h>
#include <modules/desktop_capture/desktop_capture_options.h>
#include <modules/desktop_capture/desktop_capturer.h>
#include <rtc_base/logging.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include "open3d/core/Tensor.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/utility/Console.h"
#include "open3d/visualization/webrtc_server/GlobalBuffer.h"

namespace open3d {
namespace visualization {
namespace webrtc_server {

class ImageReader {
public:
    class Callback {
    public:
        virtual void OnCaptureResult(const std::shared_ptr<core::Tensor>&) = 0;

    protected:
        virtual ~Callback() {}
    };

    ImageReader();
    virtual ~ImageReader();

    void Start(Callback* callback);

    void CaptureFrame();

    Callback* callback_ = nullptr;
};

class ImageCapturer : public rtc::VideoSourceInterface<webrtc::VideoFrame>,
                      public ImageReader::Callback {
public:
    ImageCapturer(const std::string& url_,
                  const std::map<std::string, std::string>& opts);
    virtual ~ImageCapturer();

    static ImageCapturer* Create(
            const std::string& url,
            const std::map<std::string, std::string>& opts);

    ImageCapturer(const std::map<std::string, std::string>& opts);

    bool Init();

    void CaptureThread();

    bool Start();

    void Stop();

    bool IsRunning();

    // Overide webrtc::DesktopCapturer::Callback.
    // See: WindowCapturerX11::CaptureFrame
    // build/webrtc/src/ext_webrtc/src/modules/desktop_capture/linux/window_capturer_x11.cc
    virtual void OnCaptureResult(
            const std::shared_ptr<core::Tensor>& frame) override;

    // Overide rtc::VideoSourceInterface<webrtc::VideoFrame>.
    virtual void AddOrUpdateSink(
            rtc::VideoSinkInterface<webrtc::VideoFrame>* sink,
            const rtc::VideoSinkWants& wants) override;

    virtual void RemoveSink(
            rtc::VideoSinkInterface<webrtc::VideoFrame>* sink) override;

protected:
    std::thread capture_thread_;
    std::unique_ptr<ImageReader> capturer_;
    int width_;
    int height_;
    bool is_running_;
    rtc::VideoBroadcaster broadcaster_;
};

}  // namespace webrtc_server
}  // namespace visualization
}  // namespace open3d
