# Mango Leaf Disease Prediction
マンゴーの葉の写真をアップロードすると７つの病気か健康な状態の８クラスに分類するWebアプリです。
使用した言語はpythonのみで、フロントエンドはStreamlitを使用しています。
ローカル環境で試すにはDockerfile(準備中)を元にコンテナを作成するか、すでにこのデータ用に転移学習させたmodel.pthとstreamlit_app.pyをダウンロードすれば使用可能です。
＊リソースの関係でサンプルのmodel.pthは1epochしか学習していません。
＊トレーニングに使用していない健康な葉のサンプル写真としてJPEGをリポジトリ内においてあります。

##概要


## Training用　Dataset Link:
https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset

## 参考資料:
https://blog.streamlit.io/deep-learning-apps-for-image-processing-made-easy-a-step-by-step-guide/
(https://github.com/MainakRepositor/MLDP/assets/64016811/7287fa8f-e3b0-4db2-aa62-15f700671129)
