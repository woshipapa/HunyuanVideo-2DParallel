# HunyuanVideo3DTransformer-with-Sequence-and-Data-Parallelism
This repository provides an implementation of HunyuanVideoTransformer with both single-stream and dual-stream architectures. It incorporates sequence parallelism using Ulysses' method and data parallelism based on a customized framework of accelerate and deepspeed. The implementation supports both pre-training and inference stages.



## Transformer_Hunyuan_Video
transformer_hunyuan_video.py is an extension of the original Diffusers library, which introduces support for both single-stream and dual-stream SP processing.


### Features
- Single-stream SP: Text and video inputs are processed separately before the attention mechanism. Specifically, the QKV matrices for text and video are computed independently before being passed into the attention block. This approach maintains the individual characteristics of each modality during processing.

- Dual-stream SP: Text and video inputs are concatenated as soon as they enter the block. This combined input is then used for the joint computation of the QKV matrices within the attention mechanism. This method allows for a more integrated representation of both modalities, leveraging their combined features throughout the attention process.
- Extended Utility Functions: Utilizes a custom utils library for distributed operations, including:
  -  Padding: Ensures input consistency across different tensor sizes.
  -  All2all: Distributes data across multiple devices or workers.
  -  Split and Gather: Efficient data splitting and collection operations, useful for parallel processing.




