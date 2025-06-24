import 'dart:math';

class Dictionary {
  List<dynamic> titles;
  List<List<dynamic>> data;
  Map<String, Map<dynamic, double>> dataToTokenDic = {};
  bool withNormalize;
  Dictionary({
    required this.data,
    required this.titles,
    this.withNormalize = false,
  }) {
    for (var i = 0; i < data.length; i++) {
      for (var j = 0; j < data[i].length; j++) {
        if (!dataToTokenDic.containsKey(titles[j])) {
          if (data[i][j] is String) {
            dataToTokenDic.addAll({
              titles[j]: {data[i][j]: 0.0}
            });
          } else {
            dataToTokenDic.addAll({
              titles[j]: {data[i][j]: data[i][j] + 0.0}
            });
          }
        } else {
          if (!dataToTokenDic[titles[j]]!.containsKey(data[i][j])) {
            if (data[i][j] is String) {
              dataToTokenDic[titles[j]]!.addAll(
                  {data[i][j]: dataToTokenDic[titles[j]]!.length + 0.0});
            } else {
              dataToTokenDic[titles[j]]!.addAll({data[i][j]: data[i][j] + 0.0});
            }
          }
        }
      }
    }
    if (withNormalize) {
      dataToTokenDic.forEach(
        (key, value) {
          double maxValue = value.values.reduce(max);
          double minValue = value.values.reduce(min);
          List keys = value.keys.toList();
          for (var i = 0; i < keys.length; i++) {
            double temp = value[keys[i]]!;
            temp = (temp - minValue) / (maxValue - minValue);
            value[keys[i]] = temp;
          }
        },
      );
    }
  }
  double toToken({required String title, required dynamic element}) {
    if (dataToTokenDic[title] == null) {
      throw Exception('unknown title');
    } else if (dataToTokenDic[title]![element] == null) {
      throw Exception('unknown element');
    }
    return dataToTokenDic[title]![element]!;
  }

  dynamic fromToken({required String title, required dynamic token}) {
    if (dataToTokenDic[title] == null) {
      throw Exception('unknown word');
    }
    var item;
    dataToTokenDic[title]!.forEach((key, value) {
      if (value == token) {
        item = key;
      }
    });
    return item;
  }
}
