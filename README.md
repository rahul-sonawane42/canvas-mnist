# Canvas-Mnist: Framework-Free Inference Engine

A lightweight, strictly mathematical digit recognition engine built from the metal up.

This project bypasses heavy machine learning frameworks (like TensorFlow or PyTorch) entirely. It demonstrates low-level backend system design by coupling a custom Multilayer Perceptron (MLP) written in pure NumPy with a desktop GUI and a highly controlled OpenCV data preprocessing pipeline.

The goal is not just to recognize a digit, but to demonstrate how unstructured human input is mathematically transformed, normalized, and processed through a matrix pipeline in real-time.

---

## Core Architecture

The system is decoupled into two distinct environments to separate the heavy training calculus from the low-latency inference engine:

### 1. The Math Engine (Backend)

* **Zero Dependencies:** The neural network is built using exclusively standard Python and NumPy matrix dot-products (`np.dot`).
* **Custom Abstractions:** Implements forward propagation, manual backpropagation (Chain Rule), ReLU/Softmax activations, Categorical Cross-Entropy Loss, and the Adam optimizer from scratch.
* **Deep Graph:** A 4-layer fully connected architecture: `784 (Input) → 256 (Dense + ReLU) → 128 (Dense + ReLU) → 10 (Softmax)`.
* **Edge Serialization:** After 35 epochs of training, weight and bias matrices are exported as raw binary `.npy` artifacts, allowing the frontend to load them directly into memory without framework bloat.

### 2. The Vision Pipeline (Frontend)

Dense networks lack spatial invariance. To bridge the gap between raw handwriting and rigid matrix constraints, the Pygame canvas input is intercepted by a strict computer vision pipeline before inference:

* **Extraction:** Raw 3D pixel arrays are pulled from the Pygame event loop and transposed to standard `(height, width, channels)` orientation.
* **Binary Thresholding:** Converts the grayscale drawing into high-contrast binary to eliminate anti-aliased noise.
* **Bounding-Box Isolation (`cv2.boundingRect`):** Mathematically isolates the drawn digit, stripping away all empty spatial data regardless of where it was drawn on the canvas.
* **Proportional Normalization (`cv2.resize`):** Downscales the isolated digit to fit a $20 \times 20$ bounding box using `INTER_AREA` interpolation to preserve stroke density.
* **Center-of-Mass Alignment:** Pastes the normalized digit perfectly into the dead-center of a zero-initialized $28 \times 28$ array, mirroring the exact structural curation of the original MNIST dataset.

---

## Installation & Execution

This project includes a shell script to automate the creation of an isolated Python virtual environment, preventing conflicts with global system packages.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/canvas-mnist.git
cd canvas-mnist

# 2. Make the setup script executable
chmod +x setup.sh

# 3. Run the setup and launch the engine
./setup.sh

```

### Manual Setup

If you prefer not to use the bash script:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pygame opencv-python
python app.py

```

---

## Usage & Controls

The interface is built as a responsive 960x600 dark-themed desktop dashboard displaying live probability distributions and model input rendering.

* **Mouse (Left Click):** Draw a digit on the left canvas.
* **Real-Time Inference:** The NumPy engine runs asynchronously in the background. Stop drawing for `120ms` to trigger an automatic forward pass and update the confidence bars.
* **[C] Key / Clear Button:** Wipe the canvas, reset the memory buffers, and zero out the probability distribution.

---

## Future Scope

* **Custom Convolutional Layers (CNN):** Engineering 2D convolution and pooling operations from scratch to replace the Dense architecture, natively solving the spatial invariance problem without relying entirely on OpenCV bounding boxes.
* **Multi-Digit Recognition (Sliding Window):** Upgrading the OpenCV pipeline to detect multiple distinct digit contours on the canvas simultaneously, feeding them sequentially through the network to evaluate written math (e.g., recognizing "24" instead of just "2" and "4").
---

## 📁 Repository Structure

```text
canvas-mnist/
├── app.py                 # Pygame dashboard and OpenCV vision pipeline
├── setup.sh               # Bash script for venv creation and execution
├── backend/
│   ├── train.py           # The pure NumPy training loop (Backprop, Adam, CCE)
│   ├── core/
│   │   ├── layers.py      # Custom Dense, ReLU, Softmax mathematical abstractions
│   │   └── optimizer.py   # Manual Adam optimizer implementation
│   └── models/            # Serialized .npy weight and bias matrices
└── README.md

```
