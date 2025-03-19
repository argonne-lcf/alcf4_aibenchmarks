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

## Code Access

## FOM
```bash

    $`FOM = \frac{ b_sL(H.D.W / p^3)^2 d}{T}`$,


where b_s = batch size
      L = Number of layers
      H,D,W = height, width and depth of the input image. 
      p = patch size
      d = hidden dimension of the model
      T = Time to solution. 
```
## Steps to Run



