#pragma once

#include <span>
#include <chrono>
#include <vector>
#include <iostream>
#include <random>
#include <thread>
#include <fstream>
#include <mutex>
#include <filesystem>
#include <cstddef>

#include "eigen3/Eigen"

namespace mnist {
	extern "C" {
		#include "loader.h"
	}
}

