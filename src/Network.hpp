#pragma once
#include "pch.hpp"

using Eigen::MatrixXd;

inline auto rng = std::default_random_engine();

auto sigmoid(const MatrixXd& z) {
	return 1.0/(1.0+Eigen::exp((-z).array()));
}

auto sigmoid_prime(const MatrixXd& z) {
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


