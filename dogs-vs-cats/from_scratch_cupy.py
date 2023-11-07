import cupy as np
from cupyx.scipy import signal
import matplotlib.pyplot as plt

np.random.seed(42)


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # calculates the output of the layer based on the input
    def forward_pass(self, input):
        pass

    # update the layer's parameters
    def backward_pass(self, output_gradient, learning_rate):
        pass

class Convolutional(Layer):
    # input shape: tuple, containing depth, height and width of the input
    # kernel size: int, specifying the size of the matrix inside each kernel (it's a square matrix)
    # depth: int, specifying how many kernels there are
    def __init__(self, input_shape: tuple, kernel_size: int, depth: int):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        # stride = 1, no padding
        self.output_shape = (depth,
                             input_height - kernel_size + 1,
                             input_width - kernel_size + 1)
        # 4D: multiple 3D kernels
        # depth = number of kernels
        # input_depth = depth of each kernel (depth of input)
        # the last two are to create the square matrices in each kernel
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        # initializing randomly
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward_pass(self, input):
        self.input = input
        # calculating the output (implementing the above formula)
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                # using cross correlation from scipy
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward_pass(self, output_gradient, learning_rate):
        # initializing the gradients with empty matrices
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        # gradient of the biases: how the loss function changes with respect to the biases (derivative of the error with respect to the biases)
        # the bias gradient is already given as the output_gradient parameter

        for i in range(self.depth):
            for j in range(self.input_depth):
                # gradient of the input: how the loss function changes with respect to the input to the layer
                # sensitivity of the loss with respect to changes in the input values
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
                # gradient of a kernel: how the loss function changes with respect to the individual elements of the kernel
                # sensitivity of the loss with respect to changes in the values of the kernel weights
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")

        # updating the parameters with gradient descent
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient

        return input_gradient

class Dense(Layer):
    # n_input: number of neurons in the input
    # n_output: number of neurons in the output
    def __init__(self, n_input: int, n_output: int):
        # initializing randomly
        self.weights = np.random.randn(n_output, n_input)
        self.bias = np.random.randn(n_output, 1)

    def forward_pass(self, input):
        self.input = input
        # calculating the output (implementing the above formula)
        return np.dot(self.weights, self.input) + self.bias

    def backward_pass(self, output_gradient, learning_rate):
        # gradient of a kernel: derivative of the error with respect to the weights
        weights_gradient = np.dot(output_gradient, self.input.T)
        # gradient of the input: derivative of the error with respect to the input to the layer
        input_gradient = np.dot(self.weights.T, output_gradient)
        # gradient of the biases: derivative of the error with respect to the biases = output_gradient

        # updating the parameters with gradient descent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_pass(self, input):
        return np.reshape(input, self.output_shape)

    def backward_pass(self, output_gradient, learning_rate):
        # in this case the learning_rate parameter isn't used, it's only there bc this is inherits from the layer class o.O
        return np.reshape(output_gradient, self.input_shape)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_derivative(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Sigmoid(Layer):
    def __init__(self):
        pass

    def forward_pass(self, input):
        self.input = input
        return sigmoid(self.input)

    def backward_pass(self, output_gradient, learning_rate):
        sigma_x = sigmoid(self.input)
        return np.multiply(output_gradient, (sigma_x * (1 - sigma_x)))

def relu(x):
    zeros = np.zeros(x.shape)
    return np.maximum(zeros, x)

class Relu(Layer):
    def __init__(self):
        pass

    def forward_pass(self, input):
        self.input = input
        return relu(self.input)

    def backward_pass(self, output_gradient, learning_rate):
        # elements are 0 where the condition is true, and 1 where the condition is false
        relu_derivative = np.where(self.input < 0, 0, 1)
        return np.multiply(output_gradient, relu_derivative)

class Network:
    def __init__(self, layers: list):
        self.layers = layers
        self.errors = []
        self.accuracies = []

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output

    def train(self, loss_function, loss_function_derivative, x_train, y_train, epochs=5, learning_rate=0.01):
        for i in range(epochs):
            error = 0
            correct_predictions = 0

            for x, y in zip(x_train, y_train):
                # forward
                output = self.predict(x)

                # error
                error += loss_function(y, output)

                # backward
                gradient = loss_function_derivative(y, output)
                # taking the layers in reversed order, so the gradient is propagated from the output layer to the input layer
                for layer in reversed(self.layers):
                    gradient = layer.backward_pass(gradient, learning_rate)

                # accuracy
                predicted_label = np.argmax(output)
                true_label = np.argmax(y)
                if predicted_label == true_label:
                    correct_predictions += 1

            average_error = error / len(x_train)
            self.errors.append(average_error.item())

            accuracy = correct_predictions / len(x_train) * 100
            self.accuracies.append(accuracy)

            print(f"epoch: {i + 1}/{epochs},\tloss = {average_error},\taccuracy = {accuracy}%")

    def plot_history(self):
        epochs = [(i + 1) for i in range(len(self.errors))]

        plt.figure(1)
        plt.plot(epochs, self.errors, color="red")
        plt.title("loss over epochs")
        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.figure(2)
        plt.plot(epochs, self.accuracies, color="blue")
        plt.title("accuracy over epochs")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")

        plt.show()


import os
import cv2

image_size = 128
number_of_images = 0


def load_and_preprocess_data(data_dir, image_size, sample_size = 0):
    images = []
    labels = []

    if sample_size == 0:
        sample_size = len(os.listdir(data_dir))

    # loading images
    for filename in os.listdir(data_dir):
        # reading and "preprocessing" image
        img = cv2.imread(os.path.join(data_dir, filename))
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
        # deciding the label based on the file name (example: cat.1.png)
        label = filename.split('.')[0]
        labels.append(1 if label == "cat" else 0)

    # converting lists to np arrays
    images = np.array(images)
    labels = np.array(labels)

    # normalizing
    images = images.reshape(len(images), 1, image_size, image_size)
    images = images.astype("float32") / 255

    # shuffling the data
    permutation = np.random.permutation(len(images))
    images = images[permutation]
    labels = labels[permutation]

    return images[:sample_size], labels[:sample_size]

train_data_dir = 'E:/dogs_vs_cats/train/train_small'
test_data_dir = 'E:/dogs_vs_cats/train/train_small'

x_train, y_train = load_and_preprocess_data(train_data_dir, image_size, number_of_images)
x_test, y_test = load_and_preprocess_data(test_data_dir, image_size, number_of_images)

conv1 = Convolutional(input_shape=(1, image_size, image_size),
                      kernel_size=3,
                      depth=5)

conv2 = Convolutional(input_shape=conv1.output_shape,
                      kernel_size=3,
                      depth=5)

reshaped_int = np.prod(np.array(conv2.output_shape)).item()

network = Network([
    conv1,
    Relu(),

    conv2,
    Relu(),

    Reshape(input_shape=conv2.output_shape,
            output_shape=(reshaped_int, 1)),

    Dense(n_input=reshaped_int,
          n_output=256),

    Sigmoid(),

    Dense(n_input=256,
          n_output=2),

    Sigmoid()
])


print("TRAIN:")
# train
network.train(
    loss_function=binary_cross_entropy,
    loss_function_derivative=binary_cross_entropy_derivative,
    x_train=x_train,
    y_train=y_train,
    epochs=30,
    learning_rate=0.1
)

