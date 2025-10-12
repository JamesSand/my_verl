
follow 了 weixiong 的 repo，我们使用了和他们一样的 trian set，test set

由于算力的限制，他们的 train set 有 75k 条数据，我们使用了前 10k 数据进行训练。

同时，他们的 batch size 是 512，但是我们只有 64

### 1 Prepare data

```bash
# prepare trainset
python focal_data_preprocess/preprocess_trainset.py

# prepare testset
python focal_data_preprocess/preprocess_testset.py
```

data 存储在 ./data/ 目录下

### 2 Train focal

```bash
# focal loss
bash run/focal/Qwen2.5-0.5B-gsm8k-lora32-lr5e-5-focal-loss.sh

# seq-mean-token-mean loss
bash run/focal/Qwen2.5-0.5B-gsm8k-lora32-lr5e-5-seq-token-mean.sh
```

现在控制 focal loss 的是 `loss_agg_mode` 这个参数，如果想用 focal loss，设置为 `focal-loss` 即可

现在的对比实验是开的 `loss_agg_mode=seq-mean-token-mean`，也就是先对每个 token 求 mean，再对每个 sequence 求 mean

by default 的是 'token-mean'，也就是直接对所有 token 求 mean。

`token-mean` 和 `seq-mean-token-mean` 本质上是不一样的，跟 focal loss compare 的话，`seq-mean-token-mean` 更加合理一些。

