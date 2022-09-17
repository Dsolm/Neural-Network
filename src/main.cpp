#include "pch.hpp"
#include <filesystem>
#include <nlohmann/json.hpp>

using Eigen::MatrixXd;

auto rng = std::default_random_engine();

auto sigmoid(const auto& z) {
	return 1.0/(1.0+Eigen::exp((-z).array()));
}

auto sigmoid_prime(const auto& z) {
	return sigmoid(z).array() * (1-sigmoid(z).array());
}


template <int Size>
class Network {
	public:
		Network(const std::array<int, Size>& sizes) {
			for (size_t i = 1; i < Size; i++) {
				weights[i] = MatrixXd::Random(sizes[i], sizes[i-1]);
				bias[i] = MatrixXd::Random(sizes[i], 1);
			}
		}

		MatrixXd feedforward(MatrixXd input) {
			for (int l = 1; l < Size; l++) {
				input = sigmoid((weights[l] * input) + bias[l]);	
			}
			return input;	
		};

		//maybe pass it by reference and use a mutex when shuffling if we want to save memory.
		void stochastic_gradient_descent(std::span<std::pair<MatrixXd, MatrixXd>> training_data, const int mini_batch_size, const double eta, const int epochs) {
			int n_mini_batches = training_data.size() / mini_batch_size;
			double learning_rate = eta/mini_batch_size;
			for (int epoch = 0; epoch < epochs; epoch++) {
				std::shuffle(training_data.begin(), training_data.end(), rng);

				for (int mini_batch_id = 0; mini_batch_id < n_mini_batches; mini_batch_id++) {
					//A span is like a non-owning slice of a list. It's a view into a list or some part of it.
					const std::span<std::pair<MatrixXd, MatrixXd>> mini_batch(
							training_data.begin() + mini_batch_size*mini_batch_id,
							training_data.begin() + mini_batch_size * (mini_batch_id+1)
					);
					
					std::array<MatrixXd, Size> total_change_bias;
					std::array<MatrixXd, Size> total_change_weights;
					for (int i = 1; i < Size; i++) {
						total_change_weights[i].resizeLike(weights[i]);
						total_change_weights[i].fill(0);
						total_change_bias[i].resizeLike(bias[i]);
						total_change_bias[i].fill(0);
					}

					for (const auto& [data, expected_output] : mini_batch) {
						std::array<MatrixXd, Size> weighted_sums;
						std::array<MatrixXd, Size> outputs;

						outputs[0] = data;
						//Feedforward
						MatrixXd output = data;
						for (int l = 1; l < Size; l++) {
//							std::cout << "Output cols: " << output.cols() << "rows: " << output.rows() << "\n";
//							std::cout << "Weights cols: " << weights[l].cols() << "rows: " << weights[l].rows() << "\n";
							MatrixXd weighted_sum = (weights[l] * output) + bias[l];

							weighted_sums[l] = weighted_sum;
							output = sigmoid(weighted_sum);	
							outputs[l] = output;
						}

						std::array<MatrixXd, Size> errors;

						//.array() is the type used for performing vectorized operations.
						errors.back() = (output - expected_output).array() * sigmoid_prime(weighted_sums.back()).array();	

						
						total_change_bias.back() = total_change_bias.back() + errors.back();
						total_change_weights.back() = total_change_weights.back() +
							errors.back() * (outputs[Size-2].transpose());

						for (int l = Size - 2; l >= 1; l--) {
							errors[l] = ((weights[l+1]).transpose() * errors[l+1]).array() * 
								sigmoid_prime(weighted_sums[l]).array();
							total_change_bias[l] = total_change_bias[l] + (errors[l]);
							total_change_weights[l] = total_change_weights[l] +
								errors[l]*(outputs[l-1].transpose());
						}
					}

					m_mutex.lock();
					for (int l = 1; l < Size; l++) {
						weights[l] = weights[l] - learning_rate * total_change_weights[l]; 
						bias[l] = bias[l] - learning_rate * total_change_bias[l];
					}
					m_mutex.unlock();
				}
			}
		}

		/*std::size_t getContentsByteSize() {
			std::size_t size = 0;
			for (auto& mat : weights) {
				size += mat.rows() * mat.cols() * sizeof(double);
			}
			for (auto& mat : bias) {
				size += mat.rows() * mat.cols() * sizeof(double);
			}
			return size;
		}*/
		void serialize(const std::string& output_path = "output.json") {
			nlohmann::json data;
			data["weights"] = {};

			int weight_id = 0;
			for (auto& mat : weights) {
				data["weights"][weight_id] = {};
				auto& array = data["weights"][weight_id];
				for (int i = 0; i < mat.rows(); i++) {
					for (int j = 0; j < mat.cols(); j++) {
						array.push_back(mat(i, j));
					}
				}
				weight_id += 1;
			}

			int bias_id = 0;
			data["bias"] = {};
			for (auto& mat : bias) {
				data["bias"][bias_id] = {};
				auto& array = data["bias"][bias_id];
				for (int i = 0; i < mat.rows(); i++) {
					for (int j = 0; j < mat.cols(); j++) {
						array.push_back(mat(i, j));
					}
				}
				bias_id += 1;
			}

			std::ofstream o(output_path);
			o << std::setw(4) << data << std::endl;

		}

		void deserialize(const std::string& path = "output.json") {
			std::ifstream f(path);
			nlohmann::json data;
			f >> data;

			for (unsigned int weight_id = 0; weight_id < data["weights"].size(); weight_id++) {
				const auto& weightdata = data["weights"][weight_id];
				for (int i = 0; i < weights[weight_id].rows(); i++) {
					for (int j = 0; j < weights[weight_id].cols(); j++) {
						weights[weight_id](i,j) = weightdata[i*weights[weight_id].cols() + j];
					}
				}
			}
			for (unsigned int bias_id = 1; bias_id < data["bias"].size(); bias_id++) {
				const auto& biasdata = data["bias"][bias_id];
				for (int i = 0; i < bias[bias_id].rows(); i++) {
					for (int j = 0; j < bias[bias_id].cols(); j++) {
						bias[bias_id](i,j) = biasdata[i*(bias[bias_id].cols()) + j];
					}
				}
			}
/*
			const unsigned char* cursor = buffer.begin().base();
			const unsigned char* end = buffer.end().base();
			for (auto& mat : weights) {
				cursor = Eigen::deserialize(cursor, end, mat);
			}
			for (auto& mat : bias) {
				cursor = Eigen::deserialize(cursor, end, mat);
			}
			*/
		}

		std::mutex m_mutex;
		std::array<MatrixXd, Size> weights;
		std::array<MatrixXd, Size> bias;
};

std::vector<MatrixXd> load_labels() {
	const char* path = "labels.txt";
	std::ifstream file(path);
	if (!file) {
		throw std::runtime_error("labels file not found");
	}
	std::vector<MatrixXd> labels;
	labels.reserve(376);
	for (double token; file >> token;) {
		MatrixXd tmp(1,1);
		tmp(0,0) = token;
		labels.push_back(tmp);
	}
	return labels;
}
#include <sstream>
constexpr int entries_per_input_vector = 351;
std::vector<MatrixXd> load_input_vectors(const std::string& path) {
	std::ifstream file(path);
	if (!file) {
		throw std::runtime_error("file not found: " + path);
	}
	std::vector<MatrixXd> output;
	output.reserve(376);
	for (int i = 0; i < 376; i++) {
		output.emplace_back(entries_per_input_vector, 1);
	}
	int matrix = 0;
	for (std::string line; std::getline(file, line);) {
		std::istringstream iss(line);
		int row = 0;
		for (double value; iss >> value;) {
			output[matrix](row, 0) = value;
			row++;
		}
		matrix++;
	}
	return output;
}
std::vector<MatrixXd> load_input_vectors();
std::vector<std::pair<MatrixXd, MatrixXd>> load_data() {
	auto labels = load_labels();
	auto tfbs = load_input_vectors("tfbs.txt");
	auto def_energies = load_input_vectors("def_energy.txt");

	std::vector<std::pair<MatrixXd, MatrixXd>> result;
	result.reserve(376);
	for (int i = 0; i < 376; i++) {
	//	output.push_back(std::pair<MatrixXd, MatrixXd>(MatrixXd(entries_per_input_vector*2,1),MatrixXd(1,1)));
		MatrixXd first(entries_per_input_vector*2, 1);
		for (int j = 0; j < entries_per_input_vector; j++) {
			first(j,0) = tfbs[i](j,0);
		}
		for (int j = 0; j < entries_per_input_vector; j++) {
			first(j+entries_per_input_vector,0) = def_energies[i](j,0);
		}
		result.push_back(std::pair<MatrixXd, MatrixXd>(first,labels[i]));
	}
	return result;
}
int main(int argc, char** argv) {
	std::string output_path = "output.json";
	//Load training and testing data
	auto whole_data = load_data();
	using vec_label = std::pair<MatrixXd, MatrixXd>;

	rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
	std::shuffle(whole_data.begin(), whole_data.end(), rng);
	std::span<vec_label, std::dynamic_extent> training_data(&whole_data.front(), &whole_data[((whole_data.size()*3) / 4)]);
	std::span<vec_label, std::dynamic_extent> testing_data(&whole_data[((whole_data.size()*3) / 4)], &whole_data.back());

	Network<3> network({entries_per_input_vector*2, 30, 1});

	if (argc == 1) {
		std::cout << "Invalid arguments: neuralnetwork <path_to_network_dump>/train (--output path)" << std::endl;
		return 1;
	}
	if (std::string(argv[1]) == "train") {
		//auto nthreads = std::thread::hardware_concurrency();
		size_t nthreads = 6;
		
		std::vector<std::thread> threads;
		size_t slice_size = training_data.size() / nthreads;
		for (size_t i = 0; i < nthreads; i++) {
			std::span<std::pair<MatrixXd, MatrixXd>, std::dynamic_extent> training_data_slice(training_data.begin() + (i*slice_size), slice_size);
			threads.emplace_back([&network, training_data_slice]() {
						network.stochastic_gradient_descent(training_data_slice, 10, 3, 30);	
					});
		}
		for (auto& thread : threads) {
			//Perform thread cleanup.
			thread.join();
		}
	}
	else  {
		std::filesystem::path path(argv[1]);
		if (!std::filesystem::exists(path)) {
			std::cout << "Could not find network file: " << path.string() << std::endl;
			return 1;
		}
		network.deserialize(path.string());
	}
	/////////////////////////////////////////////////////////
	//Test the network./////////////////////////////////////
	int good = 0;
	for (size_t i = 0; i < testing_data.size(); i++) {
		auto output = network.feedforward(testing_data[i].first);
		std::cout << std::round(output(0,0)) << " | " << testing_data[i].second(0,0) << '\n';
		if (std::round(output(0,0)) == testing_data[i].second(0,0)) {
			good += 1;
		}
	}

	std::cout << "Acurracy: " << (static_cast<double>(good)/testing_data.size())*100 << "%" << std::endl;
	//If we have supplied the path to an image as a command line argument, use it as testing data as well./// 
	network.serialize(output_path);
}
