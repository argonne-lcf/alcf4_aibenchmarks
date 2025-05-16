# 1T Large Language Model Training

## Overview 

​Large language models (LLMs) have become pivotal in scientific discovery, complementing their significant roles in education, healthcare, and other sectors. The Transformer architecture, particularly its self-attention mechanism, has been instrumental in the success of these models. Training such expansive models necessitates integrating various parallelization strategies, including data parallelism, tensor parallelism, and pipeline parallelism. This benchmark aims to evaluate computational capabilities in handling these Transformer models. The baseline model features 1 trillion parameters, comprising 128 Transformer layers, a hidden dimension of 25,872, and a feed-forward network (FFN) size of 98,304. ​


## Code Access
The code we provided is using Megatron-DeepSpeed framework. 

GitHub repo: https://github.com/argonne-lcf/Megatron-DeepSpeed

The dataset we used is Dolma v1.7. A preprocessed version of the dataset is can be found at /flare at Aurora: /flare/Aurora_deployment/AuroraGPT/datasets/dolma/data_v1.7_Llama2Tokenizer/

For performance testing, one can use small dataset listed here: https://github.com/argonne-lcf/Megatron-DeepSpeed/tree/main/ALCF/data-lists/aurora

## FOM

The computational complexity of the model is determined by the hidden dimension (d), sequence length (s), global batch size (b). The computation part, the main component is gemm involved in self-attention layers and MLP layers. The communication from two parts, once is from tensor parallelization (allreduce in the TP group), and the other is from data parallelism (allreduce in the DP group). There are also send/recv p2p communication from pipeline parallelism. The detailed modeling and projection report is at:  
[LLM_modeling_and_projection-LLM.pdf](https://github.com/argonne-lcf/alcf4_aibenchmarks/blob/main/LLM/Training/LLM_modeling_and_projection-LLM.pdf)

### Time to solution

The total time to solution

```math
t = \alpha \frac{bL(36sd^2 + 6s^2d)}{PP\cdot TP}  + \beta\frac{4 bLsd}{PP}
```


where $\alpha$ is the FLOP/s per GPU, and $\beta$ is the intranode allreduce bandwidth. 

The first term is from the computation, and the second term is from reduction of $[s, d]$ matrix for for 4 times. If we have data parallelism, we also have an allreduce within each DP groups. We will have $TP*PP$ allreduce concurrently in this step. If it is implemented in a efficient way, the allreduce of previous layer can be overlap with the computation of next layer. Therefore, the DP communication can be neglected. 

### FOM definition
The FOM can be defined as the ratio between the complexity of the problem and the time to solution. 


## FOM
```math
FOM = \frac{(6sd^2 + s^2d)bL}{T}
```

```bash
where s = sequence length
      L = Number of layers
      b = global batch size
      d = hidden dimension of the model
      T = time per training step. 
```

Notice that here we only consider the computation part in the complexity. The communication comes into play only because we cannot

## Steps to Run
The code relies on pytorch, megatron, deepspeed package. Example submission script on Aurora can be found ![here](./scripts/aurora/1T.sc)


