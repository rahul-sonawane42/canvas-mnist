import os
import urllib.request
import gzip
import numpy as np

from core.layers import Dense, Relu, Activation_Softmax_Loss_CCE
from core.optimizer import Optimizer_Adam


def download_and_load_mnist():
    print("Checking and loading MNIST dataset...")
    url_base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "X_train": "train-images-idx3-ubyte.gz",
        "y_train": "train-labels-idx1-ubyte.gz",
    }

    os.makedirs("data", exist_ok=True)
    dataset = {}

    for key, filename in files.items():
        filepath = os.path.join("data", filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url_base + filename, filepath)

        with gzip.open(filepath, "rb") as f:
            offset = 8 if "labels" in filename else 16
            data = np.frombuffer(f.read(), np.uint8, offset=offset)
            if "images" in filename:
                dataset[key] = data.reshape(-1, 784)
            else:
                dataset[key] = data

    print("MNIST dataset successfully parsed into NumPy arrays.")
    return dataset["X_train"], dataset["y_train"]


print("Initializing training script...")
X, y = download_and_load_mnist()
X = X.astype(np.float32) / 255.0

dense1 = Dense(784, 256)
activation1 = Relu()
dense2 = Dense(256, 128)
activation2 = Relu()
dense3 = Dense(128, 10)

loss_activation = Activation_Softmax_Loss_CCE()
optimizer = Optimizer_Adam(learning_rate=0.001, decay=1e-5)

epochs = 35
batch_size = 128
steps_per_epoch = len(X) // batch_size

print(f"Starting training loop for {epochs} epochs...")
for epoch in range(epochs):
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X_shuffled = X[keys]
    y_shuffled = y[keys]

    epoch_loss = 0
    epoch_acc = 0

    for step in range(steps_per_epoch):
        start = step * batch_size
        end = start + batch_size
        batch_X = X_shuffled[start:end]
        batch_y = y_shuffled[start:end]

        out1 = dense1.forward(batch_X)
        act1 = activation1.forward(out1)
        out2 = dense2.forward(act1)
        act2 = activation2.forward(out2)
        out3 = dense3.forward(act2)
        loss = loss_activation.forward(out3, batch_y)

        predictions = np.argmax(loss_activation.outputs, axis=1)
        accuracy = np.mean(predictions == batch_y)

        epoch_loss += loss
        epoch_acc += accuracy

        loss_activation.backward(loss_activation.outputs, batch_y)
        dense3.backward(loss_activation.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        optimizer.post_update_params()

    avg_loss = epoch_loss / steps_per_epoch
    avg_acc = epoch_acc / steps_per_epoch
    print(
        f"Epoch {epoch + 1:2d}/{epochs} | Loss: {avg_loss:.4f} | Training Accuracy: {avg_acc * 100:.2f}%"
    )

print("Training finished. Exporting model weights...")
os.makedirs("models", exist_ok=True)
np.save("models/dense1_weights.npy", dense1.weights)
np.save("models/dense1_biases.npy", dense1.biases)
np.save("models/dense2_weights.npy", dense2.weights)
np.save("models/dense2_biases.npy", dense2.biases)
np.save("models/dense3_weights.npy", dense3.weights)
np.save("models/dense3_biases.npy", dense3.biases)
print("All layer weights successfully saved to backend/models/.")
