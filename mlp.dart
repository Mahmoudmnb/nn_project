import 'dart:math';

enum AF { sigFun, rlu }

double sigFun(double y) {
  return 1 / (1 + pow(e, -y));
}

double rluFun(double y) {
  return max(0, y);
}

abstract class Node {
  double get output => 0;
  List<double> weights = [];
}

class Neuron extends Node {
  List<Node> inputs;
  List<double> weights = [];
  double bias = 1;
  double biasWeight = 1;
  double d = 0;
  double Function(double) activationFunction;
  Neuron({required this.inputs, required this.activationFunction}) {
    double min = -2.4 / inputs.length;
    double max = 2.4 / inputs.length;
    Random random = Random();
    this.biasWeight = (min + (max - min) * random.nextDouble());
    for (var _ in inputs) {
      double randomDouble = (min + (max - min) * random.nextDouble());
      weights.add(randomDouble);
    }
  }
  double get output {
    double sum = 0;
    for (var i = 0; i < inputs.length; i++) {
      sum += inputs[i].output * weights[i];
    }
    sum += bias * biasWeight;
    return this.activationFunction(sum);
  }
}

class InputNeuron extends Node {
  double input;
  InputNeuron({required this.input});
  double get output => input;
}

class Layer {
  int numOfNeurons;
  List<Node> neurons = [];
  AF activationFunction;
  Layer({required this.numOfNeurons, required this.activationFunction});
}

class MLP {
  List<Layer> layers;
  List<List<double>> inputs;
  List<double> labels;
  double lr;
  int epoch;
  MLP({
    required this.layers,
    required this.inputs,
    required this.labels,
    required this.lr,
    required this.epoch,
  }) : assert(inputs.length == labels.length) {
    layers.insert(
        0,
        Layer(
            numOfNeurons: inputs.first.length, activationFunction: AF.sigFun));
    for (var i = 0; i < layers.length; i++) {
      Layer layer = layers[i];
      List<Node> n = [];
      if (i == 0) {
        for (var i = 0; i < inputs[0].length; i++) {
          n.add(InputNeuron(input: inputs[0][i]));
        }
        layer.neurons = n;
        continue;
      }
      double Function(double) acFun;
      switch (layer.activationFunction) {
        case AF.sigFun:
          acFun = sigFun;
          break;
        case AF.rlu:
          acFun = rluFun;
          break;
        default:
          throw Exception('undefined activation function');
      }
      for (var j = 0; j < layer.numOfNeurons; j++) {
        n.add(Neuron(inputs: layers[i - 1].neurons, activationFunction: acFun));
      }
      layer.neurons = n;
    }
  }

  double feedForward({required int neuronNumber, required int layerIndex}) {
    return layers[layerIndex].neurons[neuronNumber].output;
  }

  void feedInput(int batchIndex) {
    for (var i = 0; i < inputs[batchIndex].length; i++) {
      (layers.first.neurons[i] as InputNeuron).input = inputs[batchIndex][i];
    }
  }

  void backpropagation(int batchIndex) {
    //* clear d in all neurons before backpropagation
    for (var i = 1; i < layers.length; i++) {
      for (var j = 0; j < layers[i].numOfNeurons; j++) {
        (layers[i].neurons[j] as Neuron).d = 0;
      }
    }
    //* this code for input layer
    Layer outputLayer = layers.last;
    for (var i = 0; i < outputLayer.numOfNeurons; i++) {
      double loss = getOutputLayerLoss(batchIndex, i);
      double neuronOutput =
          feedForward(neuronNumber: i, layerIndex: layers.length - 1);
      double d = neuronOutput * (1 - neuronOutput) * loss;
      (outputLayer.neurons[i] as Neuron).d = d;
    }
    //*  calculate d for hidden layers
    if (layers.length > 2) {
      for (var i = layers.length - 1; i > 1; i--) {
        for (var j = 0; j < layers[i].numOfNeurons; j++) {
          Neuron currentNeuron = layers[i].neurons[j] as Neuron;
          List<Node> neuronInputs = (layers[i].neurons[j] as Neuron).inputs;
          for (var k = 0; k < neuronInputs.length; k++) {
            Neuron n = neuronInputs[k] as Neuron;
            n.d += currentNeuron.d * currentNeuron.weights[k];
          }
        }
        Neuron firsNeuron = layers[i].neurons.first as Neuron;
        for (var l = 0; l < firsNeuron.inputs.length; l++) {
          (firsNeuron.inputs[l] as Neuron).d *=
              (firsNeuron.inputs[l] as Neuron).output *
                  (1 - (firsNeuron.inputs[l] as Neuron).output);
        }
      }
    }
    //* adjust wights
    for (var i = layers.length - 1; i > 0; i--) {
      Layer currentLayer = layers[i];
      for (var j = 0; j < currentLayer.numOfNeurons; j++) {
        Neuron currentNeuron = currentLayer.neurons[j] as Neuron;
        for (var k = 0; k < currentNeuron.weights.length; k++) {
          currentNeuron.weights[k] +=
              lr * currentNeuron.inputs[k].output * currentNeuron.d;
        }
        currentNeuron.biasWeight += lr * currentNeuron.bias * currentNeuron.d;
      }
    }
  }

  double getOutputLayerLoss(int patchIndex, int neuronNumber) {
    double actualOutput =
        feedForward(neuronNumber: neuronNumber, layerIndex: layers.length - 1);
    return labels[patchIndex] - actualOutput;
  }

  List<double> modelOutput({required int batchIndex}) {
    List<double> output = [];
    feedInput(batchIndex);
    for (var i = 0; i < layers.last.numOfNeurons; i++) {
      double o = feedForward(layerIndex: layers.length - 1, neuronNumber: i);
      output.add(o);
    }
    return output;
  }

  void train() {
    for (var e = 0; e < epoch; e++) {
      int trainAcc = 0;
      for (var i = 0; i < inputs.length; i++) {
        feedInput(i);
        backpropagation(i);
        double output = modelOutput(batchIndex: i).first;
        if (output == labels[i]) {
          trainAcc++;
        }
      }
    }
  }
}
