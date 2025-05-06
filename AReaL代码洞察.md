# AReaL 代码洞察

官方团队于 2025/04/27 发布项目[官方文档](https://deepwiki.com/inclusionAI/AReaL)

本文基于官方文档以及 [GitHub 代码](https://github.com/inclusionAI/AReaL) 尝试对该项目的昇腾支持进行洞察。

## 系统架构概览

项目主要可以分为五个子系统

- 用户接口 CLI 或 YAML
- 实验框架 管理配置以及训练过程
- 分布式训练系统 处理工作协同以及数据分发
- 模型系统 提供模型的抽象
- 评估系统 验证系统输出，主要是数学领域，还有其他

## 训练流水线架构概览

主要组成部分

- 配置创建 PPOMATHConfig
- 分布式进程管理 Master Work 和 Model Worker
- 模型 Actor, Critic, Reference, Reward
- 训练接口 PPO Actor，Critic，Reward 分别专用
- 后端 Megatron，SGLang，vllm

## 模型系统架构

提供抽象模型于各种语言模型以及后端交互

- 模型 API 模型抽象操作
- 模型后端 
  - Megatron 分布式训练
  - SGLang 生成
  - vLLM 优化推理阶段
- 模型接口 定义训练，推理，生成操作
  - PPO
  - SFT 
  - Reward
- 模型组件

### 模型系统

Model 类：包含神经网络 tokenizer 以及 metadata 等基本数据。

PipelinableEngine 类：定义一个看可以在 model 上执行的标准操作。
- train_batch
- eval_batch
- forward 执行推理
- generate 由模型生成文本

ModelBackend 类：定义初始化和管理模型执行的引擎，转化 PipelinableEngine 实例，进行分布式执行，必须实现 _initialize, destroy, save, load
- Megetron-LM 分布式训练
- SGLang 文本生成
- vLLM 加速推理
- Null Backend 实现测试模拟

ModelInterface 类：定义可以被 model 执行的高阶操作：
- SFT 
- PPO Actor 用于优化 Actor 的 Policy
- PPO Critic 用于优化 Critic 的 Value Function
- Reward 
- NullInterface 实现测试

ReaLModelConfig 类：定义模型的基本参数
- 基本的 transformer 架构参数，包括 layer，heads，dimensions
- MoE 的路由以及专家配置

常规的模型使用流程
- 建立一个模型配置
- 使用一个模型 backend 初始化模型，如 megatron
- 通过 interface 使用 model，如 ppoactor

### 模型 Backend

本部分为模型在训练和推理过程中提供执行层面的作用，其位于 model API 和模型的具体实现之间，为模型的运行提供具体的实施策略，主要包括分布式训练，效率生成以及快速推理，涉及文件 megatron.py sglang.py vllm.py

#### Megatron Backend

通过 Megatron-LM 提供分布式训练的支持，主要包括 TP，PP，DP。主要功能为处理模型并行设置，分布式优化，梯度传播以及 ckeckpoint 管理。

其中，最主要的 megatron_ctx context manager 用于配置不同并行方式的并行策略。

#### SGLang Backend

通过 SGLang Server Runtime 在优化推理阶段提供有效的文本生成能力，在 PPO 训练流程中的 token 生成中尤为重要。SGLang 服务器作为一个独立进程运行，AReaL 系统通过 HTTP 请求与其进行交互。

SGLangGenerationEngine 实现 PipelinableEngine 接口来完成 AReaL 与 SGLang 服务器的交互管理，主要包括服务器的生存，通信，文本生成请求以及权重更新。

#### vLLM Backend

通过集成 vLLM 库来提供优化的推理能力，可以提供更快的生成工作。

### 模型 Interfaces

本部分定义了在语言模型上执行的操作过程，主要为训练，推理以及生成。提供抽象类以及具体实现来解耦具体的训练推理算法与 model backend。

#### PPO Interface

主要为 Actor 和 Critic 提供模型接口。为 Actor 处理 policy model 生成文本回答，为 Critic 处理 value model 估计期望的奖励。

PPO training workflow
- 生成 actor 的回答
- reward 评估回答
- critic 计算 advantages
- 由带有 KL regularization 的 PPO 更新 actor
- 更新 critic 来提高 value estimation

#### SFT Interface

主要为模型提供训练，评估以及保存等在 SFT 场景中的应用。用于在 target tokens (不包括 prompt tokens) 上计算 loss 并且计算像 perplexity 等关键指标信息。

主要方法：
- compute_packed_sft_loss 
- train_step
- evaluate

#### Reward Interface

作用为为模型评估回答以及计算合适的奖励信号，包含数学计算任务以及编程任务。

主要方法：
- rew_inf
- math_verify_call
- code_verify_call

## 训练系统

AReaL 的训练系统用于管理各组件的协调，数据分发，PPO 训练算法的执行。

### 训练配置

在数学计算任务上的配置为 PPOMATHConfig。主要用于定义模型的角色 (actor, critic, reference, reward), interface, hyperparameters, 以及设备的分配。

PPO 工作流：
- 生成阶段 actor_gen
- 评估阶段 actor_inf ref_inf rew_inf critic_inf
- 训练阶段 actor_train critic_train

PPO 主要组件：
- PPOActorInterface
- PPOCriticInterface
- Actor Loss
- Critic Loss

主 worker 系统作用：
- 初始化模型以及 backend
- 执行数据图流
- 追踪数据训练过程
- 处理检查点，保存，恢复

数据集的配置：
- SFT 数据集使用 PromptAnswerDatasetConfig，格式为 {"prompt": str, "answer": str}
- PPO 数据集使用 PromptOnlyDatasetConfig，格式为 {"prompt": str}
- 数学和代码数据集使用 MATHCodePromptDataset ，其继承自 PyTorch Dataset，用于加载并验证数据集，输入提示词的 tokenization，训练过程中动态过滤数据，支持分布式处理

自定义配置可选项
- 模型配置：路径，类型，backend，hyperparameters
- MFC
- 分配配置：并行策略，设备网格，microbatch 细节
- 数据集：路径 最大序列长度，过滤选项，batch size

### 分布式系统架构

分布式系统架构用于编排分布式系统在多节点多 GPU 上的分配。主要包含 master worker， 分布式执行流，数据管理，名称解析，以及容错机制。

master worker 用于协调多个 model worker 并进行分布式训练：
- 编排训练过程
- 管理执行图
- 给 model worker 分发任务
- 处理检查点和评估
- 收集指标并报告

model worker 
- 加载执行模型代码
- 处理数据
- 管理 GPU 资源
- 处理参数同步
- 回应 master 请求

分布式执行流在 Controller 启动 MasterWorker 和 ModelWorkers 时开始，主要进行
- 建立不同 workers 之间的联系
- 初始化 model backends
- 加载数据集
- 建立执行图
- 建立分布式处理组

数据管理 DataManager 用于
- 在 workers 中加载和分配数据集
- 传输数据
- 缓存
- 为数据并行进行数据分片

Controller 系统时分布式架构的中心，位于实验配置层与分布式执行层之间。主要有两种，Controller 使用 ZMQ 的标准控制，RayController 专用于 Ray 的分布式执行。

- 启动配置 workers
- 监视状态
- 中断或终止 worker 执行
- 为不同的执行环境提供不同实现方式(local, Slurm, Ray)

### 评估系统

提供评估模型在数学推理以及代码生成任务中的综合能力。评估系统包括数学解题验证，代码执行验证以及自动的评估工作流等。
主要评估指标包括：
- Pass@K 评估模型在 K 次尝试的情况下解对问题的概率
- Majority Voting 在多次尝试情况下最为常见的回答会被选取

数学解题验证需要与已知的问题答案进行比较，需要处理各种数学符号以及格式，以及形式不同但值相同的数学表达式。主要方法包括
- extract_answer 解析数学表达式，从文本中提取数学答案，处理边界问题
- math_equal 计算是否两个数学表达式等价

编程验证用于评估生成的代码的功能，系统自动执行所生成代码，并进行评估。主要过程
- load_problems_with_testcase_batch 加载输入的问题以及测试用例
- 远程执行 将问题和测试用例以及代码发送到一个安全环境执行
- code_verify 结果处理
- 结论裁定 是否所有测试用例均通过

## 部署与集成

基本的软硬件需求，官方最低配置 1 节点 8 GPU 1.5B 模型，官方首选在 Ray 作为分布式框架进行部署，也可以通过 Slurm 进行
- GPU 至少 80 GB memory
- CUDA 12.5，Git LFS，Dockers 27.5.1， NVIDIA Container Toolkit

cpp CUDA 扩展用于实现高效并关键的操作：
- GAE (Generalized Advantage Estimation)