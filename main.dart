import 'mlp.dart';

void main(List<String> args) {
  MLP nn = MLP(
    labels: [0, 1, 1, 0],
    inputs: [
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0]
    ],
    layers: [
      Layer(numOfNeurons: 2, activationFunction: AF.sigFun),
      Layer(numOfNeurons: 1, activationFunction: AF.sigFun),
    ],
    lr: 0.6,
    epoch: 50000,
  );
  int batchIndex = 0;
  List<double> beforeProb = nn.modelOutput(batchIndex: batchIndex);
  print(beforeProb);
  nn.train();
  List<double> afterProb = nn.modelOutput(batchIndex: batchIndex);
  print(afterProb);
}
