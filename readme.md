# 概要
マイクラ建築画像の中で，自分の好みのモノ（イイねするもの）とそうでないものの分類をするモデル構築

# データセット
自分がイイね・保存した画像を使用
データが足りなそうなので，Pintarestから下のキーワードで検索した画像をダウンロード
[minecraft, cocricot, minecraft cafe, minecraft house, minecraft building]
Googleから以下キーワードでダウンロード(youtubeサムネ等が多いためあまり使いたくない)
[minecraft 城, cocricot]
複数枚の画像のセットは除外している

最終的に，
* favo / no_favoそれぞれ317枚ずつのデータセットを構築
* 学習：テスト = 250:67 ~= 8:2

# 試すこと＠2クラス分類
1. ResNet18のpretrainedモデルで試してみる（2クラスのクロスエントロピー）
2. 自前実装のネットワークで分類
https://qiita.com/MuAuan/items/86a56637a1ebf455e180
3. iic実装
4. SVMを利用
5. 出来たらtwitterクローラーと組み合わせて実行


# その1
損失関数はクロスエントロピーを採用．
batch=4, lr=0.001(no decay), adam@(0.5,0.99), epoch=100
parameter数：11177538
モデル名はres18
iteration: 133[Loss: 0.6171668171882629] [Acc: 0.7014925479888916]
中々なモデル精度を実現．ただ，画像サイズを244にする必要があり，もう少し大きいサイズで良かったら精度もうちょい上がったかも，

とりあえず，自作モデルでこの精度を目指してみる．


# その2
損失関数はクロスエントロピーを採用．
batch=4, lr=0.001(no decay), adam@(0.5,0.99), epoch=100
<!-- パラメータ数：188065402
パラメータ数：1561394
モデル名はmynet
MyNet(
  (conv1): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), bias=False)
  (bn1): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  (conv2): Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), bias=False)
  (bn2): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  (conv3): Conv2d(128, 128, kernel_size=(5, 5), stride=(3, 3), bias=False)
  (bn3): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  (conv4): Conv2d(128, 256, kernel_size=(5, 5), stride=(3, 3), bias=False)
  (bn4): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  (fc1): Linear(in_features=186624, out_features=1000, bias=True)
  (fc2): Linear(in_features=1000, out_features=2, bias=True)
) -->
try1
iteration: 133[Loss: 0.783205509185791] [Acc: 0.641791045665741]
各Conv層の出力はReLUを通し，fc層の出力はスロープ0.2のLeakyReLUに通す．
パラメータ数の割に余り精度が出ていないので，パラメータ削減しながらの精度向上を目指す．
ハイパラをいじるのは沼なので，固定して改善する．

try2
ランダムクロップによる位置ずれを吸収するためにMaxPoolingを施す．
全体的にCNNレイヤーの大きさを小さくする.
ランダムにグレースケール化するAugmentation追加（構造情報を少し考慮するため）
Cifarの時はカラー正規化をいれていたが，元の色合いが重要だと感じ削除
transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.4)
iteration: 133[Loss: 0.6184483766555786] [Acc: 0.6865671873092651]

try3
クロップサイズを(0.8,1.0)に調整．スペースを意識した画像とかもあるため．
ランダムに透視変換を行う．Affine変換だと歪みが大きすぎて，イイねした画像も歪んだ結果はイイねではないことがあり得たため．
iteration: 133[Loss: 0.6184482574462891] [Acc: 0.6865671873092651]

try4
過学習気味だったのでDropout層を追加．
iteration: 133[Loss: 0.6630185842514038] [Acc: 0.7238805890083313]



