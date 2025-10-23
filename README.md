# SimpleNeuralNetwork v1.0

---

A small neural network written in *pure C++* that learns the XOR function in real-time using a single hidden layer.
This project demonstrates a basic fully-connected feedforward network, backpropagation, and weight initialization using Xavier method. ANSI escape codes are used for colored console output.

---

## Features

Fully configurable via console input:
- Random or manual seed
- Number of epochs
- Learning rate
- Number of hidden neurons
- Display interval for training progress

Real-time training feedback:
- Epoch number
- XOR inputs and predicted output
- Loss per batch
- Tracks best loss and corresponding epoch
- Optionally displays final weights of the network
- ANSI-colored output for better readability
 
---

## Usage

1 Clone the repository:

```powershell
git clone https://github.com/Vladyslaa/SimpleNeuralNetwork.git
cd SimpleNeuralNetwork
```

2 Compile with GCC (or any C++ compiler, requires *C++17* or later):

```powershell
g++ -std=c++17 src/SimpleNeuralNetwork.cpp -O3 -o SimpleNeuralNetwork.exe 
```

3 Run the executable:

```powershell
./SimpleNeuralNetwork.exe
```

4 Enter the requested configuration parameters

5 Observe the network learning XOR in real-time

### Example Output

```SimpleNeuralNetwork
Epoch 10000000 | 0 XOR 0 = 0.00000089 (logit: -13.92975366, target: 0)
Epoch 10000000 | 1 XOR 0 = 0.99999803 (logit: 13.13961680, target: 1)
Epoch 10000000 | 0 XOR 1 = 0.99999857 (logit: 13.45552810, target: 1)
Epoch 10000000 | 1 XOR 1 = 0.00000210 (logit: -13.07367795, target: 0)
  Loss: 0.00000160

----------------------------------------
Neural Network Training Complete!
----------------------------------------
Best Loss: 0.00000160 at Epoch 10000000

Final XOR Evaluation:
   0 XOR 0 = 0.00000089
   1 XOR 0 = 0.99999803
   0 XOR 1 = 0.99999857
   1 XOR 1 = 0.00000210

Would you like to see final weights? (y/n):
```

---

## Notes

> Console must support ANSI escape codes for color output.

---