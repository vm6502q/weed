//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

// From https://github.com/embeddedartistry/embedded-resources/blob/master/examples/cpp/dispatch.cpp

#pragma once

#include "config.h"

#if !ENABLE_PTHREAD || !ENABLE_QUNIT_CPU_PARALLEL
#error PTHREAD or QUNIT_CPU_PARALLEL has not been enabled
#endif

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>

namespace Weed {

typedef std::function<void(void)> DispatchFn;

class DispatchQueue {
public:
    DispatchQueue()
        : quit_(false)
        , isFinished_(true)
        , isStarted_(false)
    {
        // Intentionally left blank.
    }
    ~DispatchQueue();

    // dispatch and copy
    void dispatch(const DispatchFn& op);
    // finish queue
    void finish();
    // dump queue
    void dump();
    // check if queue is finished
    bool isFinished() { return isFinished_; }

    // Deleted operations
    DispatchQueue(const DispatchQueue& rhs) = delete;
    DispatchQueue& operator=(const DispatchQueue& rhs) = delete;
    DispatchQueue(DispatchQueue&& rhs) = delete;
    DispatchQueue& operator=(DispatchQueue&& rhs) = delete;

private:
    std::mutex lock_;
    std::future<void> thread_;
    std::queue<DispatchFn> q_;
    std::condition_variable cv_;
    std::condition_variable cvFinished_;
    bool quit_;
    bool isFinished_;
    bool isStarted_;

    void dispatch_thread_handler(void);
};

} // namespace Weed
