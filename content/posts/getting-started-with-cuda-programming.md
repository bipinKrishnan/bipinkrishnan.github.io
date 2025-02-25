---
title: "Parallel programming: Writing your first CUDA and Triton Kernel"
date: 2025-02-17T14:59:24Z
draft: true
showToc: true
TocOpen: false
tags: 
  - cuda
  - openai-triton
  - parallel-programming
  - long-post
---

A lot of what I share in this post is based on the lectures and slides from [GPU mode community](https://github.com/gpu-mode) and the classic [PMPP (Programming Massively Parallel Processors)](https://www.amazon.co.uk/Programming-Massively-Parallel-Processors-Hands/dp/0123814723) book, which is very popular amoung GPU folks.

___Note__: If you are new to CUDA (Compute Unified Device Architecture), you might get confused due to the inter-changeable use of CUDA and GPU programming in the post. You can think of CUDA as a software layer that allow developers to directly program NVIDIA GPUs. So when I say CUDA programming, literally it means programming an NVIDIA GPU._

## My motivation for learning CUDA

To be honest, my motivation initially was not to learn about GPUs or programming them. I got introduced to thinking about optimizing my deep learning workload during my time at Unify, where I had the chance of contributing to the Ivy compiler which is similar to [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) if you are familiar with it. 

During several internal discussions, I came across a lot stuff that was pretty new to me at the time, including topics like operator fusion, Apache TVM, OpenAI triton and so on. Up until then I was at the other side of the spectrum -- focusing on running experiments and training machine learning models using PyTorch. I haven't given much thought into the amount of optimizations and transformations happening under the hood to make these deep learning workloads go brrr🏎️ on different hardware accelerators like GPUs and TPUs.

The idea of truly understanding the internal workings of a deep learning model and transforming that understanding into high performant code is something that strikes me and I love being at that intersection. Writing GPU kernels for your workloads is one way to achieve this. Later in this post, we will talk more about kernels and what they actually mean.

You might have a different reason for learning CUDA or you are here to explore GPU programming for fun, if so let's dive into the minimal theory required to get up and running with your first GPU kernel.

## Quick theory before getting started

We will definitely talk about the required theory as we encounter them during the coding part. But there are some jargons that are to be understood so that you have a complete sense of what's happening under the hood while you write your first CUDA program.

We will be using NVIDIA's CUDA C api to interact with and program the GPUs. It has a similar syntax to the standard C language, but it also provide additional keywords which allow our code to be run on GPUs.

### Kernel

We will encounter the term kernel as long as we write CUDA programs. In C program we have functions, similarly we can write functions that are to be executed on the GPU. These functions now have a fancy name called kernel.

This is how we usually write a normal function in C language:

```C
void simpleFunc(int *a, int *b) {
  // do some operations

}
```

If we want to execute the same function in a GPU, we simply add the `__global__` keyword infront of the function:

```C
__global__ void simpleFunc(int *a, int *b) {
  // do some operations

}
```

`__global__` is a special keyword in CUDA C to mark a function as a GPU kernel and thus allowing it to be executed on the GPU.

### Grid, block and thread

You might already know that we use GPUs to run highly parallelizable computations. We also talked about kernels earlier. To execute our kernels in GPU and get the required results, we have to launch our kernels. This is how we launch the `simpleFunc` kernel inside a CUDA program:

```C
simpleFunc <<<1, 10, 1>>> (a, b)
```

The three values between `<<<` and `>>>` are the block size, number of threads and the grid size, `a` and `b` are arguments to the `simpleFunc` kernel.

I will give you a glimpse of what's happening. # give an example