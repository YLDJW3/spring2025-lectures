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
# Tokenization
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
