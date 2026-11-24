/**
 * @file        core/perf/frame_events.cpp
 * @brief       What a frame did besides drawing (wwe2k14 Session 73; see rex/perf/frame_events.h)
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <rex/perf/frame_events.h>

namespace rex::perf::frame_events {

Counters& Get() {
  static Counters counters;
  return counters;
}

}  // namespace rex::perf::frame_events
