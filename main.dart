import 'mlp.dart';

void main(List<String> args) {
  MLP nn = MLP(
    trainLabels: [0, 1, 1, 0],
    trainInputs: [
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0]
    ],
    testInputs: [
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0]
    ],
    testLabels: [0, 1, 1, 0],
    layers: [
      Layer(numOfNeurons: 2, activationFunction: AF.sigFun),
      Layer(numOfNeurons: 1, activationFunction: AF.sigFun),
    ],
    lr: 0.6,
    epoch: 10000,
  );
  // int batchIndex = 0;
  // List<double> beforeProb = nn.modelOutput(batchIndex: batchIndex);
  // print(beforeProb);
  nn.loadWeights('w.mnb');
  nn.train();
  nn.saveWights('w.mnb');

  // List<double> afterProb = nn.modelOutput(batchIndex: batchIndex);
  // print(afterProb);
}
