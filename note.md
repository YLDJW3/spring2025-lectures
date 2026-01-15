# Lecture 1: Overview and Tokenization
## Background
1. Background
    1. Researchers are becoming **disconnected** from the underlying technology
        implement and train
        download a model and fine-tune
        prompt
    2. **Full understanding** of this technology is necessary for fundamental research
2. Small vs Large language model
    1. FLOPs spent in attention versus MLP changes with scale
    2. Emergence of behavior with scale
3. What to learn
    1. Mechanics: how things work
    2. Mindset: take scale seriously
    3. Intuition: which data and modeling decision yield good accuracy
4. Framework
    1. **accuracy = efficiency * resources**
    2. what's the best model one can build given a certain compute and data budget
## History
1. Pre-neural
    N-gram language models
2. Neural
    Seq2seq models
    Adam optimizer
    Attention mechanism
    Transformer architecture
    Mixture of experts, MoE
    Model parallelism
3. Early foundation models
    Bert
    T5
4. Scaling
    OpenAI's GPT-2
    Scaling laws
    GPT-3
5. Open models
    Hugging Face
    Llama
    Qwen
    DeepSeek
6. Levels of open
    Closed model
    Open-weight model(weight, paper, no data details)
    Open-source model
7. Frontier models
    GPT
    Gemini
    Claude
    xAI
    Llama
    DeepSeek
    Qwen
## Assignments
1. 5 Assignments
    Baiscs
    Systems
    Scaling laws
    Data
    Alignment
2. Only provided unittests and adapter interfaces
3. First make it correct, then improve accuracy and speed
## Basics
1. Goal: Get a basic version of the **full pipeline working**
2. Tokenization 
    strings <-> tokens
    BPE tokenizer
3. Architecture
    Orignal transformer
    Activation functions
    Positional encoding
    Normalization
    Placement of normalization
    MLP
    Attention
    State-space models
4. Training
    Optimizer: AdamW
    Learning rate
    Batch size
    Regularization
    Hyperparameters
5. Assignment 1
    Implement BPE tokenizer
    Implement transformer, cross-entropy loss, AdamW optimizer, traning loop
    Use raw pytorch
    Train on TinyStories and OpenWebText
## Systems
1. Goal: squeeze the most out of the hardware
2. Kernels
    organize computation to maximize utilization of GPUs by minimizing data movement
3. Parallelism
    data movement between GPUs is even slower
    collective operation
    shard across GPUs
4. Inference
    generate tokens given a prompt
    prefill: process all tokens at once
    decode: generate one token at a time
    KV caching
    baching
5. Assignment 2
    Implement a fused RMSNorm kernel in Triton
    Implement distributed data parallel training
    Implement optimizer state sharding
    **Benchmark and profile** the implementations
## Scaling laws
1. Question: Given a FLOPs budget, use a bigger model or train on more tokens?
2. Compute optimal scaling laws
    Number of training tokens = 20 * Model size
3. Assignment 3
    Traning API: hyperparamenters -> loss
    Submit training jobs under a FLOPs budget and gather data points
    Fit a scaling law
    Submit predictions for scaled up hyperparameters
## Data
1. Data composition
    Web
    Academic
    Codebase
    ...
2. Evaluation
    standardized testing
    instruction following
    full system: RAG, agents
3. Data curation
    crawl
    formats
4. Data processing
    convert HTML/PDF to text
    filtering to keep high quality data
    deduplication
5. Assignment 4
    Convert common Crawl HTML to text
    Train classifiers to filter for quality and harmful content
    Deduplication using minhash
## Alignment
1. Base model, good at completing the next token
2. Goals of alignment
    Follow instructions
    Tune the style: format, length, tone
    Safety
3. Supervised finetuning, SFT
    base model already has the skills, just a few examples to surface them
4. Learn from feedback
    Generate multiple responses
    Have user rates responses
5. Verifiers
    Formal verifiers for code, math
    Train against an LM as a judge
6. Algorithms
    Proximal Policy Optimization, PPO
    Direct Policy Optimization, DPO
    Group Relative Preference Optimization, GRPO
7. Assignment 5
    Implement supervised fine-tuning
    Implement DPO
    Implement GRPO
## Efficiency-driven deisgn decisions
1. data processing: filter bad data
2. tokenization
3. model architecture: sharing KV caches, sliding window attention
4. training
5. scaling laws
## Tokenization
1. Unicode strings <-> Integer(token) array
2. Tokenizer
    Encode: strings -> tokens
    Decode: tokens -> strings
3. **Vocabulary size** is number of possible tokens
4. Tiktokenizer
    https://tiktokenizer.vercel.app
5. Compression ratio
    num_bytes / num_tokens
6. **Character-based tokenization**
    1. each character can be converted into a code point
    2. vocabulary size is too large
    3. many characters are rare, leading to inefficiency
7. **Byte-based tokenization**
    1. UTF-8
    2. vocabulary size is 256
    3. compression ratio = 1, long sequence leading to inefficiency
8. **Word-based tokenization**
    1. vocabulary size is unbounded
    2. inefficiency
9. **Byte pair tokenization**, BPE
    ```
    Start with each byte as a token
    loop [merge_num]
        Count the numbers of each pair of tokens
        Merge the **most common pair** of adjacent tokens
    end
    ```
    ```
    indices = tokenizer.encode(string)
    reconstructed_string = tokenizer.decode(indices)
    ```
# Lecture 2: Pytorch, Resource Accounting
## Motivating question
1. How long would it take to train a 70B parameter model on 15T tokens on 1024 H100s?
    ```python
    # why 6?
    total_flops = 6 * 70B * 15T
    assert h100_flop_per_sec == 1979e12 / 2
    # what's mfu
    mfu = 0.5
    flops_per_day = h100_flop_per_sec * mfu * 1024 * 3600 * 24
    days = total_flops / flops_per_day
    ```
2. What's the largest model that can you train on 8 H100s using AdamW?
    ```python
    h100_bytes = 80e9
    # parameters, gradients, optimizer state
    bytes_per_parameter = 4 + 4 + (4 + 4)
    num_parameters = h100_bytes * 8 / bytes_per_parameter
    ```
## Memory accounting
1. tensors
2. fp32
    single precision
    32bits: 1sign + 8exponent + 23fraction
    ```python
    x = torch.zeros(m, n)
    assert get_mermory_usage(x) == m * n * 4    # bytes
    ```
3. fp16
    half precision
    16bits: 1sign + 5exponent + 10fraction
4. bf16
    16bits: 1sign + 8exponent + 7fraction
    use **same memory** as float16, has the **same dynamic range** as float32 
5. fp8
    FP8 E4M3: 1sign + 4exponent + 3fraction
    FP8 E5M2: 1sign + 5exponent + 2fraction
6. Implications on training
    Training with fp32 works, but requires lots of memory
    Training with fp8, fp16 and even bf16 is risky, and you can get instability
    Use **mixed precision training**
## Compute accouting
### tensors on GPUs
1. By default, tensors are stored in CPU memory
2. Move tensor to GPU memory
```python
y = x.to("cuda:0")
```
3. Creat a tensor on GPU
```python
z = torch.zeros(32, 32, device="cuda:0")
```
### tensor operation
1. Most tensors are created from performing operations on other tensors
2. Each operation has some memory and compute consequence
3. tensor storage
    Tensors are actually pointers to **allocated memory**
4. tensor slicing
    provide a different view of the tensor
    ```python
    x = torch.tensor([[1.,2,3],[4,5,6]])

    y = x[0] # row 0

    y = x[:, 1] #col 1

    y = x.view(3, 2) # view 2*3 matrix as 3*2 matrix

    y = x.transpose(1, 0)  # transpose
    ```
    Some views are **non-contiguous entries**, which means that further views aren't possible
5. tensor elementwise
    apply some operation to each element of the tensor and return a **new tensor** of the same shape
    ```python
    x = torch.tensor([1, 4, 9])

    x.pow(2)
    x.sqrt()
    x + x
    x * 2

    x.triu()  # takes the upper triangular part of a matrix
    ```
6. tensor matrix multiplication
    ```python
    x = torch.ones(16, 32)
    y = torch.ones(32, 2)
    z = x @ y   # 16 * 2 matrix
    ```
    perform operations for **every example in a batch** and **every token in a sequence**
    ```python
    x = torch.ones(4, 8, 16, 32)
    w = torch.ones(32, 2)
    y = x @ w   # (4, 8, 16, 2)
    ```
### tensor einops
1. **Name all the dimensions** instead of using indeices
2. jaxtyping
    ```python
    # old
    x = torch.ones(2, 2, 1, 3)
    # jaxtyping
    x: Float[torch.Tensor, "batch seq heads hidden"] = torch.ones(2,2,1,3)
    ```
3. einsum
    ```python
    z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
    z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
    ```
4. reduce
    reduce a single tensor via some operation (sum, mean, max, min)
    ```python
    y = reduce(x, "...hidden -> ...", "sum")
    ```
5. rearrange
    ```python
    x: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2,3,8)
    # total_hidden is a flatten representation of heads * hidden1
    w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4,4)
    x = rearrange(x, "...(heads hidden1) -> ...heads hidden1", heads=2)
    ```
### tensor operations flops
1. a **FLOP** is a basic operation like addition(x + y) or multiplication(x * y)
2. FLOPs vs FLOP/s
    FLOPs: floating-point operations, measure of computation done
    FLOP/s: floating-point operations per second, measure the speed of hardware