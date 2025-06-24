import 'dart:io';
import 'dart:math';

enum AF { sigFun, rlu }

double sigFun(double y) {
  return (1 / (1 + pow(e, -y)));
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
  List<List<double>> trainInputs;
  List<double> trainLabels;
  List<List<double>>? testInputs;
  List<double>? testLabels;
  double lr;
  int epoch;
  MLP({
    required this.layers,
    required this.trainInputs,
    required this.trainLabels,
    required this.lr,
    required this.epoch,
    this.testInputs,
    this.testLabels,
  }) : assert(trainInputs.length == trainLabels.length && layers.isNotEmpty) {
    layers.insert(
        0,
        Layer(
            numOfNeurons: trainInputs.first.length,
            activationFunction: AF.sigFun));
    for (var l = 0; l < layers.length; l++) {
      Layer layer = layers[l];
      List<Node> n = [];
      if (l == 0) {
        for (var i = 0; i < trainInputs.first.length; i++) {
          n.add(InputNeuron(input: trainInputs.first[i]));
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
        n.add(Neuron(inputs: layers[l - 1].neurons, activationFunction: acFun));
      }
      layer.neurons = n;
    }
  }

  double feedForward({required int neuronNumber, required int layerIndex}) {
    return layers[layerIndex].neurons[neuronNumber].output;
  }

  void feedInput(List data) {
    for (var i = 0; i < data.length; i++) {
      (layers.first.neurons[i] as InputNeuron).input = data[i];
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
          List<Node> neuronInputs = currentNeuron.inputs;
          for (var k = 0; k < neuronInputs.length; k++) {
            Neuron n = neuronInputs[k] as Neuron;
            n.d += currentNeuron.d * currentNeuron.weights[k];
          }
        }
        Neuron firsNeuron = layers[i].neurons.first as Neuron;
        for (var l = 0; l < firsNeuron.inputs.length; l++) {
          Neuron n = firsNeuron.inputs[l] as Neuron;
          n.d *= n.output * (1 - n.output);
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
    return trainLabels[patchIndex] - actualOutput;
  }

  List<double> modelOutput({required List data}) {
    List<double> output = [];
    feedInput(data);
    for (var i = 0; i < layers.last.numOfNeurons; i++) {
      double o = feedForward(layerIndex: layers.length - 1, neuronNumber: i);
      output.add(o);
    }
    return output;
  }

  void train({Function(int epoch)? callBack}) {
    for (var e = 0; e < epoch; e++) {
      double trainTotalLoss = 0;
      int trainCorrect = 0;
      double testTotalLoss = 0;
      int testCorrect = 0;
      //* trainDs
      for (var i = 0; i < trainInputs.length; i++) {
        feedInput(trainInputs[i]);
        backpropagation(i);
        double output = modelOutput(data: trainInputs[i]).first;
        double actual = trainLabels[i];
        trainTotalLoss += pow((actual - output), 2);
        int prediction = output >= 0.5 ? 1 : 0;
        if (prediction == actual) {
          trainCorrect++;
        }
      }
      trainTotalLoss /= trainInputs.length;
      double trainAccuracy = trainCorrect / trainInputs.length;
      String resultText = '';
      resultText =
          'Epoch $e: trainLoss = ${trainTotalLoss}, trainAccuracy = ${trainAccuracy}';

      //* testDs
      if (testInputs != null && testLabels != null) {
        for (var i = 0; i < testInputs!.length; i++) {
          feedInput(testInputs![i]);
          List output = modelOutput(data: testInputs![i]);
          int prediction = output.first >= 0.5 ? 1 : 0;
          double actual = testLabels![i];
          testTotalLoss += pow((actual - prediction), 2);
          if (prediction == actual) {
            testCorrect++;
          }
        }
        testTotalLoss /= testInputs!.length;
        double testAccuracy = testCorrect / testInputs!.length;
        resultText +=
            '      |      testLoss = ${testTotalLoss}, testAccuracy = ${testAccuracy}';
      }
      print(resultText);
      if (callBack != null) {
        callBack(e);
      }
    }
  }

  void saveWights(String path) {
    String text = '';
    for (var l = 1; l < layers.length; l++) {
      for (var n = 0; n < layers[l].numOfNeurons; n++) {
        for (var i = 0; i < layers[l].neurons[n].weights.length; i++) {
          text += layers[l].neurons[n].weights[i].toString() + ' ';
        }
        text += (layers[l].neurons[n] as Neuron).biasWeight.toString();
        if (n < layers[l].numOfNeurons - 1) {
          text += '\t';
        }
      }
      if (l < layers.length - 1) {
        text += '\n';
      }
    }
    File(path).writeAsStringSync(text);
  }

  void loadWeights(String path) {
    String text = File(path).readAsStringSync();
    List<String> layerStrings = text.trim().split('\n');
    for (int l = 1; l < layers.length; l++) {
      if (l - 1 >= layerStrings.length) {
        throw Exception(
            'Mismatch in saved layers and current model structure.');
      }
      List<String> neuronStrings = layerStrings[l - 1].split('\t');
      for (int n = 0; n < layers[l].numOfNeurons; n++) {
        if (n >= neuronStrings.length) {
          throw Exception(
              'Mismatch in saved neurons and model structure in layer $l.');
        }
        List<String> weightStrings = neuronStrings[n].trim().split(' ');
        Neuron neuron = layers[l].neurons[n] as Neuron;
        if (weightStrings.length != neuron.weights.length + 1) {
          throw Exception(
              'Mismatch in number of weights for neuron $n in layer $l. Expected ${neuron.weights.length + 1}, found ${weightStrings.length}.');
        }
        for (int w = 0; w < neuron.weights.length; w++) {
          neuron.weights[w] = double.parse(weightStrings[w]);
        }
        neuron.biasWeight = double.parse(weightStrings.last);
      }
    }
  }
}
