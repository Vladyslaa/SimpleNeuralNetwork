#include <iostream>

#include "../include/Math.hpp"
#include "../include/Random.hpp"
#include <string_view>
#include <string>
#include <iomanip>

#pragma region ansi_colors

constexpr std::string_view ENDL = "\033[0m\n";
constexpr std::string_view NONE = "\033[0m";

constexpr std::string_view BOLD = "\033[1m";
constexpr std::string_view FAINT = "\033[2m";
constexpr std::string_view CURSE = "\033[3m";
constexpr std::string_view NCURSE = "\033[23m";

constexpr std::string_view RED = "\033[31m";
constexpr std::string_view GREEN = "\033[32m";
constexpr std::string_view YELLOW = "\033[33m";
constexpr std::string_view BLUE = "\033[34m";
constexpr std::string_view PURPLE = "\033[35m";
constexpr std::string_view CYAN = "\033[36m";
constexpr std::string_view WHITE = "\033[37m";
constexpr std::string_view GRAY = "\033[38;5;245m";
constexpr std::string_view ORANGE = "\033[38;5;208m";

#pragma endregion

using namespace math;

static inline int string_to_number(const std::string& str) {
	if (str.empty() || !std::all_of(str.begin(), str.end(), ::isdigit)) {
		throw std::invalid_argument("Expected a non-empty positive numeric value. Received: " + str);
	}
	try {
		return std::stoi(str);
	}
	catch (const std::out_of_range&) {
		throw std::invalid_argument("Value exceeds maximum 32-bit integer (2147483647). Received: " + str);
	}
}

static inline double string_to_double(const std::string& str) {
	if (str.empty()) {
		throw std::invalid_argument("Empty argument");
	}

	try {
		std::size_t pos;
		double res = std::stod(str, &pos);
		if (pos != str.length()) throw std::invalid_argument(NULL);
		return res;
	}
	catch (const std::out_of_range&) {
		throw std::invalid_argument("Value exceeds maximum double (~1.797e+308). Received: " + str);
	}
	catch (const std::invalid_argument&) {
		throw std::invalid_argument("Expected a non-empty double value. Received: " + str);
	}
}

int main() {
	try {

#pragma region initialisation

		std::cout << std::fixed << std::setprecision(8);

		int seed;
		size_t epochs;
		double learning_rate;
		size_t hidd_neuron_count;
		size_t print_frequency;
		std::string input_buffer;

		std::cout << GRAY << std::string(61, '-') << CURSE << ORANGE 
			<< "\nSimpleNeuralNetwork v1.0" << NCURSE << GRAY << " | " << BLUE << "By " << YELLOW << "Vladysla\n" 
			<< GRAY << "A neural network learning " << GREEN << "XOR" << GRAY << " in " << PURPLE << CURSE << "real time" << ENDL
			<< RED << "\033[2m" << "Note: Colors require a console with ANSI escape code support." << ENDL
			<< GRAY << std::string(61, '-') << ENDL << ENDL;

		std::cout << GREEN << "=== Neural Network Configuration ===" << ENDL;
		std::cout << GRAY << "Please enter the following parameters:" << ENDL << ENDL;

		std::cout << CYAN << "Enter a integer seed " << CURSE << GRAY << "(press \"Enter\" to generate a random one): " << NCURSE << YELLOW;
		if (!std::getline(std::cin, input_buffer)) return 1;
		try {
			seed = string_to_number(input_buffer);
		} catch (const std::exception&) {
			seed = std::random_device{}();
		}

		std::cout << CYAN << "Enter number of epochs: " << YELLOW;
		if (!std::getline(std::cin, input_buffer)) return 1; 
		epochs = string_to_number(input_buffer);

		std::cout << CYAN << "Enter display interval in epochs: " << YELLOW;
		if (!std::getline(std::cin, input_buffer)) return 1;
		print_frequency = string_to_number(input_buffer);

		std::cout << CYAN << "Enter learning rate: " << YELLOW;
		if (!std::getline(std::cin, input_buffer)) return 1;
		learning_rate = string_to_double(input_buffer);

		std::cout << CYAN << "Enter number of hidden neurons: " << YELLOW;
		if (!std::getline(std::cin, input_buffer)) return 1;
		hidd_neuron_count = string_to_number(input_buffer);

		std::cout << GREEN << "Configuration completed successfully!" << ENDL << ENDL;

		Random::Init(seed);

		double best_loss = 1.0e300;
		size_t best_loss_epoch = 0;

		constexpr size_t input_neuron_count = 2;
		constexpr size_t output_neuron_count = 1;

		constexpr std::array<std::pair<double, double>, 4> batch = { {
			{0.0, 0.0},
			{1.0, 0.0},
			{0.0, 1.0},
			{1.0, 1.0}
		} };
		constexpr std::array<double, 4> batch_answ = { 0.0, 1.0, 1.0, 0.0 };
		
		const double xavier_hidden = math::xavier_limit((double)input_neuron_count, (double)hidd_neuron_count);
		std::vector<std::vector<double>> weight_hidd(hidd_neuron_count, std::vector<double>(input_neuron_count));
		for (size_t i = 0; i < hidd_neuron_count; ++i) {
			for (size_t j = 0; j < input_neuron_count; ++j) {
				weight_hidd[i][j] = Random::Double(-xavier_hidden, xavier_hidden);
			}
		}
		std::vector<double> bias_hidd(hidd_neuron_count, 0.0);

		const double xavier_output = math::xavier_limit((double)hidd_neuron_count, (double)output_neuron_count);
		std::vector<std::vector<double>> weight_outp(output_neuron_count, std::vector<double>(hidd_neuron_count));
		for (size_t i = 0; i < output_neuron_count; ++i) {
			for (size_t j = 0; j < hidd_neuron_count; ++j) {
				weight_outp[i][j] = Random::Double(-xavier_hidden, xavier_hidden);
			}
		}
		std::vector<double> bias_outp(output_neuron_count, 0.0);

#pragma endregion 

		for (size_t epoch = 1; epoch <= epochs; ++epoch) {
			double total_loss = 0.0;

			std::vector<std::vector<double>> acc_gradient_hidd(hidd_neuron_count, std::vector<double>(input_neuron_count, 0.0));
			std::vector<double> acc_gradient_bias_hidd(hidd_neuron_count, 0.0);
			std::vector<double> acc_gradient_outp(hidd_neuron_count, 0.0);
			double acc_gradient_bias_outp = 0.0;

			for (size_t n = 0; n < batch.size(); ++n) {
				const std::vector<double>& input = { batch[n].first, batch[n].second };
				double target = batch_answ[n];

#pragma region forward_pass

				std::vector<double> logit_hidd(hidd_neuron_count);
				std::vector<double> output_hidd(hidd_neuron_count);
				for (size_t i = 0; i < hidd_neuron_count; ++i) {
					logit_hidd[i] = weight_hidd[i] * input + bias_hidd[i];
					output_hidd[i] = math::tanh(logit_hidd[i]);
				}

				double logit_outp = weight_outp[0] * output_hidd + bias_outp[0];
				double loss = math::bce_with_logits_loss(logit_outp, target);

#pragma endregion 
#pragma region Backpropagation

				total_loss += loss;
				
				double delta_outp = math::bce_with_logits_loss_delta(logit_outp, target);

				std::vector<double> gradient_outp = output_hidd * delta_outp;
				double gradient_bias_outp = delta_outp;

				std::vector<double> delta_hidd(hidd_neuron_count);
				for (size_t i = 0; i < hidd_neuron_count; ++i) {
					delta_hidd[i] = (weight_outp[0][i] * delta_outp) * math::tanh_derivative(logit_hidd[i]);
				}

				std::vector<std::vector<double>> gradient_hidd = math::weights_gradient(delta_hidd, input);
				std::vector<double> gradient_bias_hidd = delta_hidd;

				for (size_t i = 0; i < hidd_neuron_count; ++i) {
					for (size_t j = 0; j < input_neuron_count; ++j) acc_gradient_hidd[i][j] += gradient_hidd[i][j];
					acc_gradient_bias_hidd[i] += gradient_bias_hidd[i];

					acc_gradient_outp[i] += gradient_outp[i];
				}
				acc_gradient_bias_outp += gradient_bias_outp;

				if (epoch % print_frequency == 0 || epoch == 1) {
					double probability = math::sigmoid(logit_outp);
					std::cout << GRAY << "Epoch " << ORANGE << epoch << GRAY << " | " << CYAN
						<< (int)input[0] << GRAY << " XOR " << CYAN << (int)input[1] << GRAY << " = "
						<< GREEN << probability << GRAY << " (logit: " << YELLOW << logit_outp
						<< GRAY << ", target: " << PURPLE << (int)target << GRAY << ")" << ENDL;
				}
			}

			for (size_t i = 0; i < hidd_neuron_count; ++i) {
				for (size_t j = 0; j < input_neuron_count; ++j) weight_hidd[i][j] -= learning_rate * (acc_gradient_hidd[i][j] / batch.size());
				bias_hidd[i] -= learning_rate * (acc_gradient_bias_hidd[i] / batch.size());
				weight_outp[0][i] -= learning_rate * (acc_gradient_outp[i] / batch.size());
			}
			bias_outp[0] -= learning_rate * (acc_gradient_bias_outp / batch.size());

#pragma endregion
			if (best_loss > (total_loss / batch.size())) {
				best_loss = total_loss / batch.size();
				best_loss_epoch = epoch;
			}
			
			if (epoch % print_frequency == 0 || epoch == 1) {
				std::cout << "  Loss: " << RED << total_loss / batch.size() << ENDL << ENDL;
			}
		}

		std::cout << BOLD << CYAN << std::string(40, '-') << WHITE 
			<< "\nNeural Network Training Complete!\n" << CYAN << std::string(40, '-') << ENDL;

		std::cout << GREEN << CURSE << "Best Loss: " << best_loss << " at Epoch " << best_loss_epoch << ENDL << ENDL;

		std::cout << YELLOW << BOLD << "Final XOR Evaluation:" << ENDL;
		for (auto& x : batch) {
			const std::vector<double>& input = { x.first, x.second };
			std::vector<double> logit_hidd(hidd_neuron_count);
			std::vector<double> output_hidd(hidd_neuron_count);

			for (size_t i = 0; i < hidd_neuron_count; ++i) {
				logit_hidd[i] = weight_hidd[i] * input + bias_hidd[i];
				output_hidd[i] = math::tanh(logit_hidd[i]);
			}

			double logit_outp = weight_outp[0] * output_hidd + bias_outp[0];
			double out = math::sigmoid(logit_outp);

			std::cout << "   " << GRAY << (int)x.first << " XOR " << (int)x.second << " = " << GREEN << out << ENDL;
		}

		std::cout << std::endl << CYAN << "Would you like to " << CURSE << "see final weights?" << NCURSE << " (y/n): " << ENDL;
		char choice = std::cin.get();

		if (std::tolower(choice) == 'y') {
			std::cout << std::endl << BOLD << BLUE << "Hidden Layer Weights:" << ENDL;
			for (size_t i = 0; i < weight_hidd.size(); ++i) {
				std::cout << "   ";
				for (double w : weight_hidd[i])
					std::cout << std::setw(10) << w << " ";
				std::cout << std::endl;
			}

			std::cout << std::endl << BOLD << BLUE << "Output Layer Weights:" << ENDL;
			for (double w : weight_outp[0])
				std::cout << "   " << std::setw(10) << w << std::endl;
		}

		std::cout << std::endl << GRAY << "Training session finished successfully." << ENDL;
		std::cin.get(); std::cin.get();

		return 0;
	} catch (const std::exception& e) {
		std::cerr << RED << "Error: " << e.what() << ENDL;
		std::cin.get();

		return 1;
	} catch (...) {
		std::cerr << RED << "Unknown error occurred" << ENDL;
		std::cin.get();

		return 1;
	}
}