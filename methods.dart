import 'dart:math';

import 'dictionary.dart';

Map<String, List> prepare({
  required List titles,
  required List data,
  required Dictionary dic,
}) {
  var random = Random();
  data.shuffle(random);
  List<List<double>> train_ds = [];
  List<List<double>> test_ds = [];
  List<double> train_labels = [];
  List<double> test_labels = [];
  for (var i = 0; i < (data.length * 0.8).floor(); i++) {
    List<double> temp = [];
    for (var j = 0; j < data[i].length; j++) {
      if (j == data[i].length - 1) {
        train_labels.add(dic.toToken(title: titles[j], element: data[i][j]));
      } else {
        temp.add(dic.toToken(title: titles[j], element: data[i][j]));
      }
    }
    train_ds.add(temp);
  }
  for (var i = (data.length * 0.8 + 1).floor(); i < data.length; i++) {
    List<double> temp = [];
    for (var j = 0; j < data[i].length; j++) {
      if (j == data[i].length - 1) {
        test_labels.add(dic.toToken(title: titles[j], element: data[i][j]));
      } else {
        temp.add(dic.toToken(title: titles[j], element: data[i][j]));
      }
    }
    test_ds.add(temp);
  }
  return {
    'train_ds': train_ds,
    'test_ds': test_ds,
    'train_labels': train_labels,
    'test_labels': test_labels
  };
}
