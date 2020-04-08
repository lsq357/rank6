# 思路

## 环境
- Red Hat 4.8.5-16
- python 3.7.3
- cudatoolkit  10.0.130   
- cudnn   7.6.0 

## 方案
- 根据数据传递性 数据增强
- 特征融合（预训练模型 + 腾讯词向量 + fasttext词向量）
   - 预训练模型NEZHA(https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA)
   - 预训练模型UER(https://github.com/dbiir/UER-py)
   - 腾讯词向量(https://ai.tencent.com/ailab/nlp/embedding.html)
   - fasttext词向量(https://fasttext.cc/)
- 对抗学习(FGM)
- 模型融合 两个5fold模型ensemble

## 数据和预训练模自行下载

## 运行
```shell
> cd code
> sh main.sh
```
