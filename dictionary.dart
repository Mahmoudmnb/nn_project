import 'dart:math';

class Dictionary {
  List<dynamic> titles;
  List<List<dynamic>> data;
  Map<String, Map<dynamic, double>> dataToTokenDic = {};
  Map<String, Map<double, dynamic>> tokenToDataDic = {};
  bool withNormalize;
  Dictionary({
    required this.data,
    required this.titles,
    this.withNormalize = false,
  }) {
    for (var i = 0; i < data.length; i++) {
      for (var j = 0; j < data[i].length; j++) {
        if (dataToTokenDic[titles[j]] == null) {
          if (data[i][j].runtimeType == String) {
            dataToTokenDic.addAll({
              titles[j]: {data[i][j]: 0.0}
            });
            tokenToDataDic.addAll({
              titles[j]: {0.0: data[i][j]}
            });
          } else {
            dataToTokenDic.addAll({
              titles[j]: {data[i][j]: data[i][j] + 0.0}
            });
            tokenToDataDic.addAll({
              titles[j]: {data[i][j] + 0.0: data[i][j]}
            });
          }
        } else {
          if (!dataToTokenDic[titles[j]]!.containsKey(data[i][j])) {
            if (data[i][j].runtimeType == String) {
              dataToTokenDic[titles[j]]!.addAll(
                  {data[i][j]: dataToTokenDic[titles[j]]!.length + 0.0});
              tokenToDataDic[titles[j]]!.addAll(
                  {tokenToDataDic[titles[j]]!.length + 0.0: data[i][j]});
            } else {
              dataToTokenDic[titles[j]]!.addAll({data[i][j]: data[i][j] + 0.0});
              tokenToDataDic[titles[j]]!.addAll({data[i][j] + 0.0: data[i][j]});
            }
          }
        }
      }
    }
    if (withNormalize) {
      dataToTokenDic.forEach(
        (key, value) {
          double maxValue = value.values.reduce(max);
          List keys = value.keys.toList();
          for (var i = 0; i < keys.length; i++) {
            double temp = value[keys[i]]!;
            temp /= maxValue;
            value[keys[i]] = temp;
          }
        },
      );
    }
  }
  double toToken({required String title, required dynamic element}) {
    if (dataToTokenDic[title] == null) {
      throw Exception('unknown word');
    }
    return dataToTokenDic[title]![element] ?? -1;
  }

  dynamic fromToken({required String title, required dynamic token}) {
    if (tokenToDataDic[title] == null) {
      throw Exception('unknown word');
    }
    if (withNormalize) {
      double maxValue = tokenToDataDic[title]!.keys.reduce(max);
      return tokenToDataDic[title]![token * maxValue];
    }
    return tokenToDataDic[title]![token];
  }
}
