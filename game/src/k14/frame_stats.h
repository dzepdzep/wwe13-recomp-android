// WWE '13 port: the guest frame counter (VdSwap) that the D3D layer reads.
#pragma once
#include <atomic>
#include <cstdint>
namespace wwe2k14 {
inline std::atomic<uint64_t> g_guest_frame{0};
}
