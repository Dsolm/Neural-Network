#pragma once

#include <span>
#include <chrono>
#include <vector>
#include <iostream>
#include <random>
#include <thread>
#include <fstream>

#include <eigen3/Eigen/Dense>

namespace mnist {
	extern "C" {
		#include "loader.h"
	}
};

