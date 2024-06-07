# multi_layer_perceptron_with_l1l2
L1，L2正則化項を追加した多層パーセプトロンの実装  
Implementation of a multilayer perceptron with additional L1 and L2 regularization terms

# APIs
~~~ python
MLP(n_input, n_comp, n_output, batch_size=256, n_epoch=100, class_weight=None, lambda_l1=0.01, lambda_l2=0.01, seed=42)
~~~
L1，L2及び中間層の数を事前に設定する多層パーセプトロン  
- n_input, n_comp, n_output : 入力変数の数，中間層のノード数，出力変数の数（目的変数の数）
- class_weight : 目的変数のリストを入力するとsk-learnにおけるclass_weight=balancedと等価になる（e.g. class_weight=y_train）
- lambda_l1, lambda_l2 : L1，L2正則化項の強さ（大きいほど強く制約する）
- seed : 生命、宇宙、そして万物についての究極の疑問の答え

~~~ python
AutoMLP(n_input, n_comp, n_output, batch_size=256, n_epoch=100, class_weight=None, trial_size=100, seed=42)
~~~
TPEに基づき，L1，L2及び中間層の数を探索する多層パーセプトロン（n_compの値は何でもよい）  

- n_input, n_comp, n_output : 入力変数の数，中間層のノード数，出力変数の数（目的変数の数）
- class_weight : 目的変数のリストを入力するとsk-learnにおけるclass_weight=balancedと等価になる（e.g. class_weight=y_train）
- trial_size : L1，L2，中間層のノード数を探索する回数
- seed : 生命、宇宙、そして万物についての究極の疑問の答え

`if __name__ == "__main__":` のサンプルプログラム参照  

~~~ python
fit(X, y, verbose=False, val_X=None, val_y=None)
~~~
- X : 訓練データ
- y : 訓練データの目的変数
- verbose : 損失と正答率を表示
- val_X : 評価データ
- val_y : 評価データの目的変数

return  
- history : Dict型，各epochにおける損失関数と正答率と各層の重みを返す
~~~ python
predict(X, y)
~~~
- X : テストデータ
- y : テストデータの目的変数
  
return  
- results : Dict型，予測ラベル (pred)と事後確率 (pred_prob)，正解ラベル (true)を返す  

~~~ python
turning_params(X, y, val_X, val_y)
~~~
`AutoMLP`のみの関数，L1，L2及び中間層の数を探索する
- X : 訓練データ
- y : 訓練データの目的変数
- val_X : 評価データ
- val_y : 評価データの目的変数

## For example
~~~ python
model = MLP(n_input, 8, 2)
history = model.fit(X_train, y_train)
results = model.predict(X_test, y_test)


model = AutoMLP(n_input, 0, 2)
model.turning_params(X_train, y_train, X_test, y_test) 
history = model.fit(X_train, y_train, val_X=X_test, val_y=y_test)
results = model.predict(X_test, y_test)
~~~
