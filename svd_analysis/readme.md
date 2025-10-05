
## SVD Script Description

这里有一些我的脚本，按照用到的顺序介绍

1. merge_fsdp_models.sh merge 训练完的 fsdp checkpoint into 一个大的 checkpoint folder
2. merge_lora.py 把 lora 和 weight matrix 合到一起。参考 https://github.com/volcengine/verl/pull/1821#issuecomment-3150776590 进行的实现。Need to manually change the lora\_alpha, since it will be automatically set to 0, where we use 32 in our experiments. 
3. svd_decompose.py Choose which weights you want to decompose with SVD. Will save U S V as npy files. 
4. svd_analysis.py analyze the SVD results. 
5. qwen2.5-0.5b-svd.log store the final analysis of the results. 

## My Analysis

Zhizhou: My analysis to the results。下边贴了一些实验结果，完整的结果在 qwen2.5-0.5b-svd.log 里边

1. 如果看相对误差的话，pissa 在第二到第四个 eigen value 上的变化比 LoRA 更大，但是也只是大一点点 （这里 Model 0 是 Qwen2.5-0.5B, Model 1 是 LoRA， Model2 是 Pissa）

[Eigen change vs. Model 0] Model 1 / Model 0 (length = 64 ):
[2.4820916223688982e-05, -1.020063677970029e-06, 0.0, 1.7067220596800325e-06, 1.021659318212187e-05]

[Eigen change vs. Model 0] Model 2 / Model 0 (length = 64 ):
[2.0556153685902245e-05, -8.99510723684216e-06, -9.731116733746603e-06, 1.3558958016801625e-05, 6.974789357627742e-06,]


2. Principal Angle 的角度来看，pissa 调不过 LoRA。我还没太想明白为什么这个事情会发生

3. 如果我们从 row-wise 考虑 SVD 分解的 U 矩阵构成的 space，会发现 pissa 会把 row-wise 弄的更加不平均。

关注这个是因为 U matrix 里边 column wise 的向量模长都是 1，但是 row wise 可以反应哪个维度在 U matrix 里边更加重要。GPT 把这个解释成每个 row 有自己的 “能量”

=== U-space row-norms (||U_k[i,:]||_2) and pairwise changes ===
( 0 , 1 ) |Δrow-norm|: min = 2.9802322387695312e-08 , mean = 1.8143295164918527e-05 , max = 8.07344913482666e-05 , signed_mean = -1.6873569563813362e-07
( 0 , 2 ) |Δrow-norm|: min = 4.470348358154297e-08 , mean = 1.605669967830181e-05 , max = 0.00010311603546142578 , signed_mean = 3.9887214597911225e-07

### Merged Model on Huggingface

model 被我上传到这里了，有 merge 之前的 (base and adaptor) 版本的，还有 merge 之后的 (merged) 版本的

https://huggingface.co/JameSand/models 

如果是 load unmerge 版本的用下边这个指令

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained(model_dir)  
pmodel = PeftModel.from_pretrained(model, lora_adapter_dir) 
```

如果是 load merged 的用这个指令
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_dir)  
```

### 怎么把 pissa 接到 LuckyRL 框架里边

抱歉我实在没时间把这个接进去了。我的这个 commit 的 diff 展示了怎么把 pissa 换进去，只需要加一个 lora init 的参数就行了

https://github.com/JamesSand/my_verl/commit/c9fb9f893a6557f4ae8c8ac47019eb0bfbf95ecf


这个是参照 peft 里边的 pissa 的 readme 做的

https://github.com/huggingface/peft/blob/24aebeec21ec44704980f46db0af1e988935de17/examples/pissa_finetuning/README.md 
