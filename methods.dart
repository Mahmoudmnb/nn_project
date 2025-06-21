import 'dart:io';
import 'dart:math';

import 'dictionary.dart';

Map<String, List> prepare({
  required List titles,
  required List data,
  required Dictionary dic,
}) {
  data.shuffle(Random());
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
  for (var i = (data.length * 0.8).floor(); i < data.length; i++) {
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

void saveDs(
    {required String path, required List<List> data, required List labels}) {
  String text = '';
  for (var i = 0; i < data.length; i++) {
    for (var j = 0; j < data[i].length; j++) {
      text += data[i][j].toString();
      if (j < data[i].length - 1) {
        text += ' ';
      }
    }
    text += '\t';
    text += labels[i].toString();
    if (i < data.length - 1) {
      text += '\n';
    }
  }
  File(path).writeAsStringSync(text);
}

Map<String, dynamic> loadDs({required String path}) {
  String text = File(path).readAsStringSync();
  List<List<double>> data = [];
  List<double> labels = [];
  List lines = text.split('\n');
  for (var i = 0; i < lines.length; i++) {
    List t = lines[i].split('\t');
    String inputs = t.first;
    labels.add(double.parse(t[1].toString()));
    List rowData = inputs.split(' ');
    List<double> temp = [];
    for (var j = 0; j < rowData.length; j++) {
      temp.add(double.parse(rowData[j].toString()));
    }
    data.add(temp);
  }
  return {'data': data, 'labels': labels};
}
