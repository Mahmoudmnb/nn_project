import 'elman.dart';
import 'methods.dart';

void main(List<String> args) {
  // final input = File('Breast_Cancer.csv').readAsStringSync();
  // final rows = const CsvToListConverter().convert(input);
  // List titles = rows.first;
  // rows.removeAt(0);
  // Dictionary dic = Dictionary(data: rows, titles: titles, withNormalize: true);

  // Map preparedDs = prepare(titles: titles, data: rows, dic: dic);
  // List<List> trainInputs = preparedDs['train_ds'];
  // List trainLabels = preparedDs['train_labels'];
  // List<List> testInputs = preparedDs['test_ds'];
  // List testLabels = preparedDs['test_labels'];
  // saveDs(path: 'train_ds.ds.mnb', data: trainInputs, labels: trainLabels);
  // saveDs(path: 'test_ds.ds.mnb', data: testInputs, labels: testLabels);

  // Map trainDs = loadDs(path: 'train_ds.ds.mnb');
  // Map testDs = loadDs(path: 'test_ds.ds.mnb');
  // List<List<double>> trainInputs = trainDs['data'];
  // List<double> trainLabels = trainDs['labels'];
  // List<List<double>> testInputs = testDs['data'];
  // List<double> testLabels = testDs['labels'];
  // int epoch = 10;
  // MLP nn = MLP(
  //   trainLabels: trainLabels,
  //   trainInputs: trainInputs,
  //   testInputs: testInputs,
  //   testLabels: testLabels,
  //   layers: [
  //     Layer(numOfNeurons: 30, activationFunction: AF.sigFun),
  //     Layer(numOfNeurons: 1, activationFunction: AF.sigFun),
  //   ],
  //   lr: 0.01,
  //   epoch: epoch,
  // );
  // nn.loadWeights('30_neuron.w.mnb');
  // nn.train(
  //   callBack: (e) {
  //     // if (e > 3 && e % (epoch / 3).floor() == 0) {
  //     //   nn.lr /= 10;
  //     // }
  //     nn.saveWights('30_neuron.w.mnb');
  //   },
  // );

  //! Elman

  // final input = File('NetFlix.csv').readAsStringSync();
  // final rows = const CsvToListConverter().convert(input, eol: '\n');
  // List titles = rows.first;
  // rows.removeAt(0);
  // Dictionary dic = Dictionary(data: rows, titles: titles, withNormalize: true);

  // Map preparedDs = prepareElmanDs(titles: titles, data: rows, dic: dic);
  // List<List> trainInputs = preparedDs['train_ds'];
  // List trainLabels = preparedDs['train_labels'];
  // List<List> testInputs = preparedDs['test_ds'];
  // List testLabels = preparedDs['test_labels'];
  // saveDs(path: 'elman_train_ds.ds.mnb', data: trainInputs, labels: trainLabels);
  // saveDs(path: 'elman_test_ds.ds.mnb', data: testInputs, labels: testLabels);

  Map trainDs = loadDs(path: 'elman_train_ds.ds.mnb');
  Map testDs = loadDs(path: 'elman_test_ds.ds.mnb');
  List<List<double>> trainInputs = trainDs['data'];
  List<double> trainLabels = trainDs['labels'];
  List<List<double>> testInputs = testDs['data'];
  List<double> testLabels = testDs['labels'];
  int epoch = 1000;
  Elman nn = Elman(
    trainLabels: trainLabels,
    trainInputs: trainInputs,
    testInputs: testInputs,
    testLabels: testLabels,
    layers: [
      Layer(numOfNeurons: 15, activationFunction: AF.sigFun),
      Layer(numOfNeurons: 1, activationFunction: AF.sigFun),
    ],
    lr: 3,
    epoch: epoch,
  );
  nn.loadWeights('elman_15_neuron.w.mnb');
  nn.train(
    callBack: (e) {
      // if (e > 3 && e % (epoch / 3).floor() == 0) {
      //   nn.lr /= 10;
      // }
      nn.saveWights('elman_15_neuron.w.mnb');
    },
  );
}
