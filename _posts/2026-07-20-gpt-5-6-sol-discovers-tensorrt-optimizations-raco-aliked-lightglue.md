---
title: "GPT-5.6 Sol: Discovering TensorRT Optimizations for RaCo-ALIKED-LightGlue+"
excerpt: "GPT-5.6 Sol finds exact TensorRT graph optimizations for RaCo-ALIKED-LightGlue+, delivering a 2.67x median speedup over FP16 torch.compile while preserving near-parity in matching quality."
categories:
  - Blog
tags:
  - model optimization
  - performance tuning
  - image matching
  - computer vision
  - raco
  - aliked
  - lightglue
  - onnx
  - onnxruntime
  - nvidia
  - tensorrt
  - gpt-5.6
  - llm
classes: wide
read_time: false
header:
  teaser: /assets/images/posts/gpt-5-6-sol-tensorrt/header.jpg
  overlay_image: /assets/images/posts/gpt-5-6-sol-tensorrt/header.jpg
  overlay_filter: 0.5
  show_overlay_excerpt: false
  caption:
    "Tokyo Station - Marunouchi"
  actions:
    - label: Models & Code
      url: https://github.com/fabio-sim/LightGlue-ONNX
mathjax: true
---

[RaCo](https://github.com/cvg/RaCo) (3DV 2026[^1]) is a new learned keypoint detector that was trained to be rotationally robust. Alongside it, the authors also trained a more performant LightGlue+ matcher that inherits this strong rotational invariance by matching ALIKED descriptors at RaCo keypoints.

![Rotating matching animation](/assets/images/posts/gpt-5-6-sol-tensorrt/marunouchi-animation.webp "Rotating matching animation"){: .align-center}

This article follows a series of blog posts[^2][^3] on accelerating inference and performance optimization of image matching pipelines. In this post specifically, we cover the below main contributions:
- An ONNX-exportable implementation of the RaCo-ALIKED-LightGlue+ pipeline that supports true end-to-end parallel batching and dynamic input sizes, including full compatibility with TorchDynamo instead of the legacy TorchScript exporter.
- How [GPT-5.6 Sol](https://openai.com/index/gpt-5-6/)[^4] identified performance bottlenecks across model components and discovered ingenious optimizations that, when benchmarked against the fastest off-the-shelf execution provider (TensorRT FP16), achieved up to a **2×** speedup.
    - To emphasize, TensorRT is already faster than the baseline (FP16 *torch.compile*) by about 1.3×.
    - Sol **doubled** that.

![Summary Speedup](/assets/images/posts/gpt-5-6-sol-tensorrt/executive-summary-speedup.svg "Summary Speedup"){: .align-center style="width: 90%;"}

- An evaluation of the optimized model across a wide range of input shapes, demonstrating substantial performance gains while preserving near-parity in matching quality with the original implementation.

All models and code are available at [fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX).

# Introduction

![RaCo-ALIKED-LightGlue+ Pipeline](/assets/images/posts/gpt-5-6-sol-tensorrt/raco-aliked-lightglue-pipeline.svg "RaCo-ALIKED-LightGlue+ Pipeline"){: .align-center}

The general structure of an image matching pipeline with LightGlue consists of:

1. A **keypoint detector**, which identifies salient locations in an image and represents them as $$(x,y)$$ coordinates alongside a confidence or ranking score.
2. A **descriptor extractor**, which produces a vector embedding for each detected keypoint, describing the local image appearance around that location.
3. A **matcher**, which receives the keypoints and descriptors extracted from two images and predicts which locations correspond to one another.

In conventional learned feature extractors such as SuperPoint or DISK, keypoint detection and descriptor extraction are performed by the same model. RaCo-ALIKED instead separates these responsibilities: RaCo decides **where *best* to look**, while ALIKED describes **what is there**. The resulting keypoints and descriptors are then passed to LightGlue+, which performs contextual matching between the two images.

Concretely in our batched implementation, given a tensor containing interleaved image pairs of shape $$(2B, 3, H, W)$$, RaCo detects and ranks a fixed number $$N$$ of keypoints for each image:

- keypoints: $$(2B, N, 2)$$
- scores: $$(2B, N)$$

Internally, RaCo first proposes a larger candidate pool of keypoints (say, $$N_{cand}$$) for the ranker to select from. Based on the ranking, we then select the top $$N$$ keypoints, discarding the rest. We implement this using a simple heuristic $$N_{cand} = P \cdot N$$, where $$P$$ is a constant factor, and find that $$P=2$$ is an acceptable tradeoff between efficiency and quality. However, as we will see later on, this tradeoff was effectively eliminated by one of Sol's optimizations.
{: .notice--warning}

ALIKED independently computes dense multi-scale feature maps from the same images. Rather than using ALIKED's own detector, descriptors are sampled from these feature maps at the coordinates selected by RaCo:

- descriptors: $$(2B, N, D)$$

where $$D=128$$ is the ALIKED descriptor dimension. LightGlue+ then processes the interleaved keypoints and descriptors and returns:

- matches: $$(S, 3)$$
- match scores: $$(S,)$$

where the first column of `matches` identifies the image pair, and the remaining columns identify the corresponding keypoint indices.

---

# Challenges

## Implementing True Batching / Parallelization

Similar to our previous work on extractor-matcher acceleration[^2], the original implementations of RaCo and ALIKED were not designed with parallel batching in mind. That is, batching was *'simulated'* by an explicit for-loop over the batch dimension, which, under Python, does not pose a problem. However, when such a model graph is traced and captured for export, the resulting graph no longer generalizes to arbitrary batch sizes, and instead is hardcoded to the specific batch size used during tracing. This is a problem for inference, where we want to be able to process arbitrary numbers of image pairs in a single forward pass.

Fortunately, using our prior batched implementations of SuperPoint, DISK, and LightGlue as references, it was straightforward for GPT-5.6 Sol to one-shot refactor the RaCo and ALIKED modelling code to support true parallel batching. The resulting exported pipeline can thus accept arbitrary batches of image pairs as input.

## Dynamic Input Shapes under TorchDynamo Export

We were glad to find out that the Dynamo-based exporter in PyTorch has matured to the point where the entire image matching pipeline can be safely exported to ONNX. In particular, after specifying the correct constraints on the input dimensions, dynamic shapes worked flawlessly.

## Deformable Convolutions

As anyone who has tried to export ALIKED or any other model that uses deformable convolutions to ONNX knows, this is a very common blocker. And although since then ONNX has added a `DeformConv` operator as of opset 19 as well as improvements to registering this operator during PyTorch-to-ONNX export, to our knowledge, TensorRT still does not natively recognize this operator unless an additional plugin is used. This is particularly troublesome when ONNX Runtime dispatches the model to execution providers, as we want the entire model to run solely on TensorRT without any graph breaks or fallbacks to other providers.

Interestingly, to solve this problem, Sol proposed a simple solution to keep the model portable: algebraically rewrite the deformable convolution as equivalent standard `GridSample` operations, allowing the model to execute on every execution provider out of the box.

## FP16 Numerical Stability

Another common headache practitioners face when converting/executing models in FP16 is the risk of overflow in certain operations. As expected, the RaCo-ALIKED-LightGlue+ pipeline is no exception. When executed by TensorRT with FP16 enabled, the pipeline would collapse completely, producing zero matches for any pair of images.

In the past, debugging such issues would have been a nightmarish and tedious process of bisecting the model graph in search of the offending operation. However, with the help of Sol, it was capable of quickly pinpointing the exact place in the model where the overflow occurred and fixed it.

**More Detailed Traceback**: The problem originated from the conversion of flattened integer pixel indices into two-dimensional coordinates. The exported graph first cast the indices to floating point, performed division followed by `Floor`, and finally cast the result back to an integer. TensorRT lowered the floating-point computation to FP16, where sufficiently large flattened indices exceeded the maximum finite representable value and overflowed to infinity. Sol traced the failure to this sequence and replaced it with direct integer floor division. This preserved the coordinate calculation entirely in integer space and exported to a standard ONNX `Div` operator.
{: .notice--warning}

---

# Evaluation Methodology & Results

We evaluate both inference performance and matching quality on the MegaDepth-1500[^5] test set. The benchmark contains 1,500 image pairs divided across five overlap intervals, providing a broad range of viewpoint, scale, illumination, and scene-depth changes.

To evaluate how the pipeline behaves under different workloads, we benchmark all combinations of the following square input resolutions and numbers of keypoints per image:

- Image resolutions: $$512^2$$, $$768^2$$, $$1024^2$$, and $$1280^2$$
- Keypoints per image: $$512$$, $$1024$$, $$1536$$, $$2048$$, $$2560$$, $$3072$$, and $$3584$$

This produces 28 configurations for each model variant, with every configuration evaluated over all 1,500 image pairs.

We compare three relevant implementations:

1. The half-precision `torch.compile()` implementation, used as the PyTorch performance baseline.
2. The naive ONNX model executed using the ONNX Runtime TensorRT execution provider with FP16 enabled.
3. The Sol-optimized ONNX model executed using the same provider configuration.

We define latency as the median GPU execution time taken to process each image pair over the entire dataset, excluding any data loading, preprocessing, compilation, or engine initialization time. We measure steady-state inference latency after warmup.

We evaluate matches against MegaDepth’s ground-truth camera geometry, reporting correspondence-weighted epipolar precision at 1, 3, and 5 px Sampson-error thresholds, alongside match counts and zero-match rates as coverage indicators, without any inlier filtering such as RANSAC or MAGSAC.

**Reproducibility**: All measurements cover the complete RaCo-ALIKED-LightGlue+ pipeline and were collected on an Intel Core i9-12900HX CPU and an NVIDIA GeForce RTX 4080 Laptop GPU with 12 GB of VRAM. The benchmark environment used Python 3.12, `torch==2.13.0+cu130`, `torchvision==0.28.0`, `onnx==1.22.0`, `onnxscript==0.7.1`, `onnxruntime-gpu==1.27.0`, `tensorrt-cu13==10.16.1.11`, `polygraphy==0.50.3`, CUDA 13.0, and cuDNN 9.20.0.48. The half-precision `torch.compile()` model is used as the PyTorch baseline.
{: .notice--info}

## Naive TensorRT Performance

Before looking at individual graph optimizations, it is helpful to first compare the naive ONNX Runtime TensorRT deployment against the half-precision `torch.compile()` baseline across the full benchmark matrix. The heatmap below reports the median inference speedup for every combination of input resolution and keypoint count.

![Naive TensorRT Speedup Heatmap](/assets/images/posts/gpt-5-6-sol-tensorrt/preoptimization-speedup-heatmap.svg "Naive TensorRT Speedup Heatmap"){: .align-center}

Even without any further optimization, TensorRT already outperforms the compiled PyTorch baseline across most tested configurations, with a median speedup of 1.38×. It is especially effective for high-workload regimes (larger image resolutions and higher keypoint numbers).

![Naive TensorRT Match Quality](/assets/images/posts/gpt-5-6-sol-tensorrt/preoptimization-match-quality.svg "Naive TensorRT Match Quality"){: .align-center}

We also verified that this deployment path remained comparable to the baseline. The figure above highlights three representative workloads: an image-heavy configuration, a sequence-heavy configuration, and the combined stress case.

Across all three workloads, TensorRT preserves correspondence-weighted epipolar precision very closely, with only small differences at the 3 px and 5 px thresholds and a larger but still modest drop at 1 px in the image-heavy case, from 77.59% to 76.30%. The main tradeoff is slightly reduced coverage: TensorRT retains 98.5%, 97.3%, and 94.9% of the baseline mean match count across the three configurations, while zero-match pairs increase modestly under the higher-keypoint workloads. This suggests that TensorRT primarily removes marginal correspondences rather than degrading the geometric accuracy of the matches it keeps.

### Profile Analysis

TensorRT is already fast, so the next question is where the remaining latency comes from. To answer this, we profiled the naive FP16 TensorRT engine at the combined-stress configuration of $$1280^2$$ input resolution and $$3584$$ keypoints per image. The per-layer profile revealed that the computational balance of the exported graph differed substantially from what one might expect from the original model architecture.

![Naive TensorRT Latency Breakdown](/assets/images/posts/gpt-5-6-sol-tensorrt/preoptimization-latency-breakdown.svg "Naive TensorRT Latency Breakdown"){: .align-center}

One might expect that LightGlue, being a transformer-based model, would dominate the workload with heavy multi-head attention layers. However, the profile shows that this is far from the case. In fact, the matcher only accounts for ~10% of the total latency, while the pipeline is predominantly bottlenecked by the detector - notably, the `TopK` operator, `Conv`, and `BatchNorm` layers.

**Why is `TopK` so expensive?**: RaCo applies it directly to the **entire full-resolution detector score map**. At $$1280^2$$, that means scanning roughly **1.64 million scores per image** to retain the top $$3584$$ keypoints. Although conceptually simple, a GPU implementation must read the complete score map, repeatedly compare and reduce candidates, and maintain a large set of provisional winners. TensorRT's direct `TopK` kernel implementation scales poorly with **both** the number of candidates and the number of winners.
{: .notice--warning}

---

# Optimizations Discovered by GPT-5.6 Sol

![Optimization Latency Waterfall](/assets/images/posts/gpt-5-6-sol-tensorrt/optimization-latency-waterfall.svg "Optimization Latency Waterfall"){: .align-center}

## Hierarchical Chunked TopK

![Hierarchical Chunked TopK](/assets/images/posts/gpt-5-6-sol-tensorrt/chunked-topk-animation.webp "Hierarchical Chunked TopK"){: .align-center}

Sol replaced the single global selection with an exact two-stage formulation. The flattened score map is divided into chunks of 65,536 elements, `TopK` is applied independently within each chunk, and the resulting local candidates are concatenated before a final `TopK` selects the global winners:

$$
\text{Global TopK}
\quad\longrightarrow\quad
\text{Chunk TopK} \rightarrow \text{Concatenate} \rightarrow \text{Final TopK}
$$

This transformation preserves the selected values and indices, apart from potentially different ordering between exactly tied boundary values. Any element belonging to the global top $$K$$ must also belong to the top $$K$$ of its own chunk; otherwise, that chunk alone would contain at least $$K$$ elements with larger scores. The final selection therefore operates on a much smaller candidate set without changing the detector's effective output.

The smaller selection problems were substantially more favorable to TensorRT. At $$1280^2$$ resolution and $$3584$$ output keypoints, the detector-selection cost fell from approximately 39.5 ms to 4.1 ms, reducing median end-to-end inference latency from 130.49 ms to 94.64 ms. The optimization also improved the image-heavy and sequence-heavy configurations, confirming that the gain was not limited to a single input shape.

## Folding Batch Normalization into Convolutions

This should normally be a standard optimization done automatically, but it is unknown why TensorRT did not perform this fusion tactic in the naive engine. Sol explicitly folded the batch normalization parameters into the preceding convolution weights and biases, eliminating the need for separate `BatchNorm` layers.

$$
W' = \frac{\gamma W}{\sqrt{\sigma^2+\epsilon}},
\qquad
b' = \beta + \frac{\gamma(b-\mu)}{\sqrt{\sigma^2+\epsilon}}
$$

## Simplifying the RaCo Detector Graph (Native SELU + Logit-Space Processing)

This was really two closely related graph simplifications:
- First, the in-place SELU implementation exported into a fragmented collection of primitive operations. Replacing it with a standard non-in-place SELU allowed ONNX to emit the native `Selu` operator, which TensorRT could fuse more effectively.
- Second, RaCo originally converted detector logits to probabilities before non-maximum suppression and ranking. Because softmax is monotonic, candidate ordering can be performed directly in logit space without evaluating a full-resolution softmax:

$$
x_i > x_j
\iff
\operatorname{softmax}(x)_i > \operatorname{softmax}(x)_j
$$

## Bounding the Ranker Selection

RaCo’s learned ranker originally performed a full `argsort` over all detector candidates and then sliced the first $$K$$ entries. Sol replaced this with a simple `topk`, which avoids fully sorting candidates that will ultimately be discarded.

---

# End-to-End Results

Putting all the optimizations together, we now re-evaluate the Sol-optimized ONNX Runtime TensorRT deployment against the baseline implementation. The heatmap below reports the median speedup for every combination of input resolution and keypoint count.

![Optimized TensorRT Speedup Heatmap](/assets/images/posts/gpt-5-6-sol-tensorrt/postoptimization-speedup-heatmap.svg "Optimized TensorRT Speedup Heatmap"){: .align-center}

Across all 28 configurations, the optimized pipeline substantially outperforms the half-precision `torch.compile()` baseline, with a median inference speedup of **2.67×**. The largest gains appear at higher keypoint counts - particularly for $$512^2$$ inputs, where the speedup reaches 3.36× at 3584 keypoints - while larger image sizes show consistently strong improvements of roughly 2.5-3.1× across most of the range. This demonstrates that the graph optimizations improve both image-heavy and sequence-heavy workloads rather than targeting only a narrow operating point.

![Optimized TensorRT Match Quality](/assets/images/posts/gpt-5-6-sol-tensorrt/postoptimization-match-quality.svg "Optimized TensorRT Match Quality"){: .align-center}

Across the three representative workloads, the optimized pipeline remains close to the baseline in correspondence-weighted epipolar precision while retaining at least **98.6%** of its mean match count. The largest precision difference appears at the strict 1 px threshold in the combined-stress configuration, where precision decreases from 74.16% to 73.11%, while the 3 px and 5 px metrics remain nearly unchanged. Coverage is similarly preserved in aggregate, although zero-match pairs increase modestly for the high-keypoint workloads, indicating that the optimized graph primarily affects a small number of difficult image pairs rather than overall matching accuracy.

# Performance-Quality Tradeoffs

> Which resolution and keypoint count should a user choose?

Although every optimized configuration outperforms its corresponding baseline, increasing image resolution and keypoint count still introduces the usual latency-quality tradeoff. To identify the most useful operating points, we compare each configuration by its median inference latency and geometrically correct match yield, retaining only those for which no other configuration is both faster and more accurate.

![Pareto Frontier of Optimized Configurations](/assets/images/posts/gpt-5-6-sol-tensorrt/postoptimization-pareto-frontier.svg "Pareto Frontier of Optimized Configurations"){: .align-center}

The Pareto frontier contains eight non-dominated operating points, spanning from the low-latency $$512^2,\ K=1024$$ configuration at 11.9 ms to the highest-yield $$1024^2,\ K=3584$$ configuration at 48.4 ms. Increasing the keypoint budget at $$512^2$$ provides particularly favorable gains, reaching 765 correct correspondences at only 20.4 ms. Moving to $$768^2,\ K=3584$$ offers the strongest quality-throughput balance, increasing the mean correct-match count to 979 at 31.1 ms. By comparison, $$1024^2,\ K=3584$$ adds only 13 additional correct correspondences while increasing latency by more than 17 ms (diminishing returns), and every $$1280^2$$ configuration is dominated by a faster configuration with equal or better correspondence yield. In practice, $$512^2,\ K=3072$$ or $$K=3584$$ are strong balanced defaults, while $$768^2,\ K=3584$$ is the preferred operating point when matching quality is prioritized.

# Conclusion

In this work, we built an ONNX-exportable RaCo-ALIKED-LightGlue+ pipeline with true end-to-end batching, dynamic input shapes, and portable deformable convolutions. The resulting model runs entirely through ONNX Runtime's TensorRT execution provider, while a targeted integer-space fix prevents FP16 overflow from silently collapsing its matches.

In particular, many of the optimizations were discovered creatively by GPT-5.6 Sol, which, from our experience, would not have been possible or would be more difficult to elicit from prior generations of models like GPT-5.5 and GPT-5.4, highlighting the step-change in capabilities of this model.

Across all 28 tested configurations, the optimized pipeline achieved a median **2.67×** speedup over FP16 `torch.compile()` while preserving near-parity in epipolar precision and at least **98.6%** of the baseline mean match count across the representative workloads.

# References

[^1]: Abhiram Shenoi, Philipp Lindenberger, Paul-Edouard Sarlin, Marc Pollefeys; "RaCo: Ranking and Covariance for Practical Learned Keypoints" in *International Conference on 3D Vision (3DV)*, 2026. Available: [URL](https://arxiv.org/abs/2602.15755)
[^2]: Fabio Milentiansen Sim, "Accelerating LightGlue Inference with ONNX Runtime and TensorRT", 2024. Available: [URL](/blog/accelerating-lightglue-inference-onnx-runtime-tensorrt/)
[^3]: Fabio Milentiansen Sim, "Float8 (FP8) Quantized LightGlue in TensorRT with NVIDIA Model Optimizer: up to ~6× faster and ~68% smaller engines", 2026. Available: [URL](/blog/fp8-quantized-lightglue-tensorrt-nvidia-model-optimizer/)
[^4]: OpenAI, "GPT-5.6: Frontier intelligence that scales with your ambition", 2026. Available: [URL](https://openai.com/index/gpt-5-6/)
[^5]: Zhengqi Li, Noah Snavely; "MegaDepth: Learning Single-View Depth Prediction From Internet Photos" in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 2041-2050.
