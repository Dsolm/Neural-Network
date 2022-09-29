#include "pch.hpp"
#include "Network.hpp"

std::vector<double> load_labels(const std::string& path) {
	std::ifstream file(path);
	if (!file) {
		throw std::runtime_error("labels file not found");
	}
	std::vector<double> labels;
	for (double token; file >> token;) {
		labels.push_back(token);
	}
	return labels;
}

std::vector<std::vector<double>> load_input_vectors(const std::string& path) {
	std::ifstream file(path);
	if (!file) {
		throw std::runtime_error("file not found: " + path);
	}
	size_t known_size = 0;
	std::vector<std::vector<double>> vectors;
	for (std::string line; std::getline(file, line);) {
		std::istringstream iss(line);
		size_t row = 0;
		std::vector<double> value_vector;
		if (known_size) {
			value_vector.resize(known_size);
			for (double value; iss >> value;) {
				value_vector[row] = value;
				row++;
			}
		}
		else {
			for (double value; iss >> value;) {
				value_vector.push_back(value);
				row++;
			}
			known_size = value_vector.size();
		}
		assert(row == known_size);
		vectors.push_back(value_vector);
	}
	return vectors;
}

std::vector<std::pair<MatrixXd, MatrixXd>> load_data_shuffled() {
	auto labels = load_labels("data/all_labels.txt");
	std::vector<std::vector<double>> tfbs_and_energies;
	{
		//hi ha un balanç correcte
		auto tfbs = load_input_vectors("data/all_tfbs.txt");
		auto def_energies = load_input_vectors("data/all_def_energy.txt");

		assert(tfbs.size() == def_energies.size());
		assert(labels.size() == tfbs.size());
		assert(tfbs[0].size() == def_energies[0].size());

		rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
		for (size_t i = 0; i < def_energies.size(); i++) {
			std::vector<double> concat_data;
			concat_data.insert(concat_data.end(),
									std::make_move_iterator(tfbs[i].begin()),
									std::make_move_iterator(tfbs[i].end())
			);
			concat_data.insert(concat_data.end(),
									std::make_move_iterator(def_energies[i].begin()),
									std::make_move_iterator(def_energies[i].end())
			);
			std::shuffle(concat_data.begin(), concat_data.end(), rng);
			tfbs_and_energies.push_back(std::move(concat_data));
		}
	}

	std::vector<std::pair<MatrixXd, MatrixXd>> result;
	result.reserve(tfbs_and_energies.size());
	for (size_t i = 0; i < tfbs_and_energies.size(); i++) {
	//	output.push_back(std::pair<MatrixXd, MatrixXd>(MatrixXd(entries_per_input_vector*2,1),MatrixXd(1,1)));
		MatrixXd first(tfbs_and_energies.size(), 1);
		//Mix descriptors
		for (size_t j = 0; j < tfbs_and_energies[0].size(); j++) {
			first(j,0) = tfbs_and_energies[i][j];
		}
		MatrixXd label_matrix (1,1);
		label_matrix(0,0) = labels[i];
		result.push_back(std::pair<MatrixXd, MatrixXd>(first,label_matrix));
	}
	return result;
}

std::vector<std::pair<MatrixXd, MatrixXd>> load_data() {
	auto labels = load_labels("data/labels_chrII_new.txt");
	//hi ha un balanç correcte
	auto tfbs = load_input_vectors("data/tfbs_chrII_new.txt");
	auto def_energies = load_input_vectors("data/def_energy_chrII_new.txt");

	assert(tfbs.size() == def_energies.size());
	assert(labels.size() == tfbs.size());
	assert(tfbs[0].size() == def_energies[0].size());

	std::vector<std::pair<MatrixXd, MatrixXd>> result;
	result.reserve(tfbs.size());
	for (size_t i = 0; i < tfbs.size(); i++) {
	//	output.push_back(std::pair<MatrixXd, MatrixXd>(MatrixXd(entries_per_input_vector*2,1),MatrixXd(1,1)));
		MatrixXd first(def_energies[0].size() + tfbs[0].size(), 1);
		//Mix descriptors
		for (size_t j = 0; j < tfbs[0].size(); j++) {
			first(j,0) = tfbs[i][j];
		}
		for (size_t j = 0; j < def_energies[0].size(); j++) {
			first(j+tfbs[0].size(),0) = def_energies[i][j];
		}
		MatrixXd label_matrix (1,1);
		label_matrix(0,0) = labels[i];
		result.push_back(std::pair<MatrixXd, MatrixXd>(first,label_matrix));
	}
	return result;
}

std::vector<std::pair<MatrixXd, MatrixXd>> load_data_energy_only() {
	auto labels = load_labels("data/all_labels_new.txt");
	//hi ha un balanç correcte
	//auto tfbs = load_input_vectors("data/all_tfbs_new.txt");
	auto def_energies = load_input_vectors("data/all_def_energy_new.txt");

	/*
	 * Amb els tts la predicció és pitjor perque els nfr que trobem al terminating site presenten un perfil menys definit que els del tss. Hi ha barreja de dades (el inici d'un es troba amb el final de un altre alguns cops)
	 * */
	//assert(tfbs.size() == def_energies.size());
	//assert(labels.size() == tfbs.size());
	//assert(tfbs[0].size() == def_energies[0].size());

	std::vector<std::pair<MatrixXd, MatrixXd>> result;
	result.reserve(def_energies.size());
	for (size_t i = 0; i < def_energies.size(); i++) {
	//	output.push_back(std::pair<MatrixXd, MatrixXd>(MatrixXd(entries_per_input_vector*2,1),MatrixXd(1,1)));
		MatrixXd first(def_energies[0].size() + def_energies[0].size(), 1);
		//Mix descriptors
		//for (size_t j = 0; j < tfbs[0].size(); j++) {
		//	first(j,0) = tfbs[i][j];
		//}
		for (size_t j = 0; j < def_energies[0].size(); j++) {
			//first(j+tfbs[0].size(),0) = def_energies[i][j];
			first(j,0) = def_energies[i][j];
		}
		MatrixXd label_matrix (1,1);
		label_matrix(0,0) = labels[i];
		result.push_back(std::pair<MatrixXd, MatrixXd>(first,label_matrix));
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

	int n_input_rows = whole_data[0].first.rows();
	Network<3> network({n_input_rows, 30, 1});

	if (argc == 1) {
		std::cout << "Invalid arguments: neuralnetwork <path_to_network_dump>/train (--output path)" << std::endl;
		return 1;
	}
	if (std::string(argv[1]) == "train") {
		//auto nthreads = std::thread::hardware_concurrency();
		size_t nthreads = 4;
		
		std::vector<std::thread> threads;
		size_t slice_size = training_data.size() / nthreads;
		for (size_t i = 0; i < nthreads; i++) {
			std::span<std::pair<MatrixXd, MatrixXd>, std::dynamic_extent> training_data_slice(training_data.begin() + (i*slice_size), slice_size);
			threads.emplace_back([&network, training_data_slice]() {
						network.stochastic_gradient_descent(training_data_slice, 10, 3, 30);
						//passar-se el mínim
						//learning rate massa gran
						//fer proves
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
	//NFR
	int true_positive = 0;
	int false_positive = 0;
	//Non NFR
	int true_negative = 0;
	int false_negative = 0;
	for (size_t i = 0; i < testing_data.size(); i++) {
		auto output = network.feedforward(testing_data[i].first);
		std::cout << std::round(output(0,0)) << " | " << testing_data[i].second(0,0) << '\n';
		int rounded_output = std::round(output(0,0));
		int expected_output = testing_data[i].second(0,0);
		if (rounded_output == expected_output) {
			if (rounded_output == 1) true_positive += 1;
			if (rounded_output == 0) true_negative += 1;
			good += 1;
		}
		else if (rounded_output < expected_output) {
			false_negative += 1;
		}
		else if (rounded_output > expected_output) {
			false_positive += 1;
		}
	}

	std::cout << "Good: " << (static_cast<double>(good)/testing_data.size())*100 << "%" << std::endl;

	std::cout <<  "true_positive: " << true_positive << std::endl;
	std::cout <<  "false_positive: " << false_positive << std::endl;
	std::cout <<  "true_negative: " << true_negative << std::endl;
	std::cout <<  "false_negative: " << false_negative << std::endl;
	std::cout << "Precision: " << (((double)true_positive)/((double)(true_positive + false_positive))) << std::endl;
	std::cout << "Negative Predictive Value: " << (((double)true_negative)/((double)(true_negative + false_negative))) << std::endl;
	std::cout << "Sensitivity: " << (((double)true_positive)/((double)(true_positive + false_negative))) << std::endl;
	std::cout << "Specificity: " << (((double)true_negative)/((double)(true_negative + false_positive))) << std::endl;
	std::cout << "Accurracy: " << (((double)true_positive + true_negative)/((double)(true_positive + true_negative + false_positive + false_negative))) << std::endl;

	network.serialize(output_path);
}

//Confusion matrix
//https://rapidminer.com/wp-content/uploads/2022/06/Confusion-Matrix-1.jpeg

//Entrenar amb un arxiu i predir els altres
//Les zones on hi ha un nucleosoma acostumen a presentar una periodicitat en la seqüència del DNA.
/////Introduïr com a descriptor la seqüència genòmica
