# 概要
マイクラ建築画像の中で，自分の好みのモノ（イイねするもの）とそうでないものの分類をするモデル構築

# 方法
自分がイイね・保存した画像を使用
データが足りなそうなので，Pintarestから下のキーワードで検索した画像をダウンロード
[minecraft, cocricot, minecraft cafe, minecraft house, minecraft building]
Googleから以下キーワードでダウンロード(youtubeサムネ等が多いためあまり使いたくない)
[minecraft 城, cocricot]
複数枚の画像のセットは除外している

# 試すこと＠2クラス分類
1. 自前実装のネットワークで分類(VGG16よりも層数は少なくする)
https://qiita.com/MuAuan/items/86a56637a1ebf455e180
2. EfficientNet実装
2. EfficientNetのファインチューニングで分類・VGG16も．
3. SVMを利用
4. 出来たらtwitterクローラーと組み合わせて実行


# 試すこと＠異常検知


# 実装１
損失関数はクロスエントロピーを採用