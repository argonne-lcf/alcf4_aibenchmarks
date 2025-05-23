# Vision Transformer

## Overview 
A 3D Vision Transformer (3D ViT) is an extension of the Vision Transformer
(ViT) architecture designed to process 3D data, such as videos, and volumet-
ric images (e.g. medical scans like CT or MRI). These kind of vision models
are specifically important to ALCF users targeting APS volumetric data such
as processing BCDI data or spatio-temporal inputs from climate applications.
Specifically, we stress-test the backbone model of VIT which also generalizes to
newer VIT variants. We use a transformer based no-mask encoder to processes
the sequence of 3D patch embeddings using self-attention mechanisms and feed-
forward layers. We can extend the work to include the decoder architecture in the future. 

## Steps to Run
Instructions are in the readme of submodule [here](https://github.com/Eugene29/Megatron-DeepSpeed_ViT/blob/794237fd851b4c268b4b07cd87685306fef986ba/README.md).
## FOM
The detailed modeling and projection report is at: [ViT-Performance_Modeling.pdf](https://github.com/argonne-lcf/alcf4_aibenchmarks/blob/main/ViT/ViT_modeling_and_projection.pdf)
 ```math
 FOM = \frac{ b_sL(N^2 d+6 d^2 N)}{T} 
```
where 
 ```math
N = H.D.W / p^3
```

```bash
where b_s = global batch size
      L = Number of layers
      N = sequence length
      H,D,W = height, width and depth of the input image. 
      p = patch size
      d = hidden dimension of the model
      T = Time to solution. 
```

## Link to experimental results 
https://argonnedoe-my.sharepoint.com/:x:/r/personal/vsastry_anl_gov/Documents/3D_ViT_benchmarks_experiments.xlsx?d=w697311c7877e492ca135d882927524a7&csf=1&web=1&e=QpfU9s



