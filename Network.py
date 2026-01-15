import numpy as np
import time

def loss_grad(output, y):
    num_classes = output.shape[1]
    if len(y.shape) == 1:
        y = np.eye(num_classes)[y]
    return output - y

def cross_entropy(outputs, results):
    total, num_classes = outputs.shape
    if len(results.shape) == 1:
        results = np.eye(num_classes)[results]
    epsilon = np.exp(-15)
    outputs = np.clip(outputs, epsilon, 1-epsilon)
    summed = -1 * np.sum(results * np.log(outputs))
    return summed / total

def accuracy(outputs, results):
    outputs = np.argmax(outputs, axis=-1)

    total = outputs.shape[0]
    comparison = outputs == results

    correct = np.sum(outputs == results)

    acc = correct / total
    # print(f'Correct: {correct}, Total: {total}, Accuracy: {acc}')
    return acc

class Network:
    def __init__(self, layers=None):
        self.layers = layers or []

    def add_layer(self, layer):
        self.layers.append(layer)

    def insert_layer(self, layer, idx=-1):
        self.layers.insert(idx, layer)

    def pop_layer(self, idx=-1):
        return self.layers.pop(idx)

    def forward(self, x):
        for layer in self.layers:
            # start = time.time()
            x = layer.forward(x)
            # end = time.time() - start
            # print(layer)
            # print('Time:', end)
            # print()
            
        return x

    def backward(self, output, dout):
        for i in range(len(self.layers)-1, -1, -1):
            dout = self.layers[i].backward(dout)

    def epoch(self, x, results, batch_size=1):
        total, _, _, _ = x.shape
        losses, accs = 0, 0
        for i in range(0, len(x), batch_size):
            end = min(len(x), i + batch_size)
            size = end - i
            x_batch = x[i:end, :, :, :]
            result = results[i:end]
            output = self.forward(x_batch)
            losses += cross_entropy(output, result) * size
            accs += accuracy(output, result) * size
            dout = loss_grad(output, result)
            self.backward(output, dout)

        loss = losses / total
        acc = accs / total
        return loss, acc

    def train(self, x, results, batch_size=1, epochs=3):
        for i in range(epochs):
            loss, acc = self.epoch(x, results, batch_size)
            print(f'Epoch {i+1}: Accuracy = {acc}, Loss = {loss}')

    def evaluate(self, x, results):
        outputs = self.forward(x)
        acc = accuracy(outputs, results)
        loss = cross_entropy(outputs, results)
        return acc, loss

    def predict(self, x):
        probs = self.forward(x)
        out = np.argmax(probs, axis=-1)
        return out

    def __repr__(self):
        out = ''
        for layer in self.layers:
            out += str(layer) + '\n'
        return out