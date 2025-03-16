
---

# 从零开始训练自己的CLM(Causal Language Model)

本项目提供了一整套基于 GPT-2 架构的因果语言模型（CLM）训练源代码，涵盖了四个主要阶段：
- **预训练**：从头开始大规模语料的无监督学习；
- **监督微调（SFT）训练**：在特定任务数据上进行有监督的微调；
- **直接偏好优化（DPO）训练**：利用用户偏好信息对模型进行优化；
- **LoRA 训练**：采用低秩适配（LoRA）技术，利用极少量参数对模型进行高效微调。

整个代码设计以实战为导向，支持在本地环境下运行与调试，旨在帮助广大学习和研究 CLM 模型的朋友，通过 GPT-2 模型训练这一具体案例，构建和训练出属于自己的因果语言模型。

## 目录结构

以下为项目的一个典型目录结构示例：

```
├── data                # 数据预处理及存储目录
├── model               # 模型权重保存与加载目录
├── train               # 训练、调试以及辅助脚本
│   ├── pretrain.py     # 预训练脚本
│   ├── sft.py          # 监督微调训练脚本
│   ├── dpo.py          # DPO 训练脚本
│   └── lora.py         # LoRA 训练脚本
├── config              # 各训练阶段的配置文件
├── requirements.txt    # Python 依赖包列表
└── README.md           # 项目说明文档
```

## 特性

- **预训练模块**  
  实现基于 GPT-2 架构的无监督预训练，帮助模型学习语言结构和语义信息。

- **SFT 训练模块**  
  提供有监督微调方案，使模型在特定任务或领域上获得更优表现。

- **DPO 训练模块**  
  采用直接偏好优化的方法，将用户或专家偏好融入模型训练流程，从而进一步提升模型响应的质量与人性化。

- **LoRA 训练模块**  
  采用低秩适配技术，在不改变模型主体参数的前提下，仅对少量参数进行更新，大幅降低资源消耗与训练成本，同时兼顾模型性能与适应性。

- **本地运行与调试**  
  项目支持在本地环境下完整地运行和调试，方便学习和研究人员深入理解每个训练阶段的细节。

## 环境要求与依赖

- **Python 版本**：Python 3.9+
- **依赖库**：项目所需的第三方库已列在 `requirements.txt` 中。建议使用如下命令安装：
  ```bash
  pip install -r requirements.txt
  ```
- **硬件要求**：推荐使用支持 CUDA 的 GPU 来加速训练过程（也可在 CPU 上运行，但速度较慢）。

    RTX 4060 Ti 16G X 1

## 使用说明

1. **克隆项目仓库**  
   在终端中执行以下命令克隆仓库并进入项目目录：
   ```bash
   git clone https://github.com/6zeus/dorami.git
   cd dorami
   ```

2. **安装依赖**  
   执行以下命令安装所有所需依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. **数据准备**  
   将预训练和微调所需的数据放入 `data` 目录中，并根据需要调整 `config` 中的配置文件参数。

4. **运行训练脚本**  
   根据需要选择不同的训练阶段：
   - **预训练**  
     ```bash
     python train/pretrain.py --config config/pretrain.json
     ```
   - **监督微调（SFT）训练**  
     ```bash
     python train/sft.py --config config/sft.json
     ```
   - **DPO 训练**  
     ```bash
     python train/dpo.py --config config/dpo.json
     ```
   - **LoRA 训练**  
     ```bash
     python train/lora.py --config config/lora.json
     ```
   脚本中的命令行参数和配置项可根据具体实验要求进行调整。

5. **调试与测试**  
   项目支持在本地运行时进行断点调试和参数验证，方便用户通过逐步执行来观察模型训练的中间效果与指标。

6. **训练成果**

- [预训练](https://huggingface.co/lucky2me/Dorami)

- [SFT](https://huggingface.co/lucky2me/Dorami-Instruct)

- [DPO](https://huggingface.co/lucky2me/Dorami-Chat)
 
- [LoRA](https://huggingface.co/lucky2me/Dorami-Chat-LoRA)

## 参数配置

- 每个训练阶段均提供详细的参数配置，请先确认 `config` 目录下对应配置文件中的参数设置是否符合当前实验环境要求。
- 为获得最佳训练效果，可依据数据规模、模型复杂度及训练资源灵活调整批次大小、学习率等参数。

## 贡献

欢迎对此项目提出改进建议、反馈问题或直接提交 Pull Request。您的每一份贡献都将帮助更多学习和研究 CLM 模型的伙伴！

## 许可协议

本项目采用 [Apache 2.0](LICENSE) 许可协议。详情请参见 LICENSE 文件。

## 联系方式

如果在使用或学习过程中有任何疑问，欢迎在 GitHub 上提交 Issue 进行讨论。

---

通过本项目，我们希望广大 CLM 学习者与研究者能够借助 GPT-2 模型训练这一个实例，更好地理解和掌握因果语言模型的核心训练流程，进而开发出属于自己的定制化模型！

---

更多与 CLM 模型训练相关的高级话题、调优策略或案例分享，后续将陆续更新。如果你对模型优化、跨领域迁移等问题感兴趣，也可以在项目中留言讨论。Happy coding!

---