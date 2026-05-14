#pragma once

namespace lidar_net {

#define AT_UNLIKE(x) __builtin_expect((x), 0)
#define AT_LIKELY(x) __builtin_expect((x), 1)

}  // namespace lidar_net
