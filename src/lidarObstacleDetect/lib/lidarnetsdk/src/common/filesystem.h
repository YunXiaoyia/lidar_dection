#pragma once

#ifdef __GNUC__
#include <features.h>
#if __GNUC_PREREQ(8, 0)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif
#endif
