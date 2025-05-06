# AReaL 社区洞察


## AReaL 背景

AReaL (Ant Reasoning Reinforcement Learning) 项目是由蚂蚁技术研究院和清华大学交叉信息院吴翼团队，联合发布的开源强化学习训练框架，并公开全部数据和完成可复现的训练脚本。AReaL 源自开源项目 ReaLHF (该项目与蚂蚁集团关系不大，并已经 archived)，旨在让每个人都能用强化学习轻松训练自己的推理模型和智能体。

AReaL 团队的核心成员均来自于蚂蚁研究院强化学习实验室以及交叉信息研究院吴翼团队，项目也借鉴了大量优秀的开源项目，比如 DeepScaleR、SGLang、QwQ、Open-Reasoner-Zero、OpenRLHF、veRL、Light-R1 和 DAPO。作为国内第一个完整开源（数据、代码、模型、脚本全开源）的强化学习项目团队，AReaL 希望能真正实现 AI 训练的普惠。

[AReaL Github](https://github.com/inclusionAI/AReaL)

[ReaLHF Github(Archived)](https://github.com/openpsi-project/ReaLHF)

团队于 2025/04/27 发布项目[官方文档](https://deepwiki.com/inclusionAI/AReaL)

团队于 2024/06/20 于Arxiv上传论文 [v1](https://arxiv.org/abs/2406.14088v1)

另于 2025/04/24 上传论文 [v2](https://arxiv.org/abs/2406.14088v2)

## 版本发布策略

目前项目的版本较少，项目名称尚未确定，社区没有明确说明更新周期。

| 版本号 | 项目名称 | 发布时间 | 时间间隔 |
| ------- | ------- | ------- | ------- |
| v0.2  | A-ReaL-Boba   | 2025/03/31   |   \ |
| v0.1  | AReaL   | 2025/02/24   | 5个月 |
| v0.3  | ReaLHF   | 2024/09/05   | \ |

## 团队主要人物

该项目目前尚未形成较大社区，亦暂时没有官方网站，暂通过 GitHub 以及所发表的论文分析团队情况。从 ReaLHF 时期开始，主要的代码提交者主要就为 MEI Zhiyu 和 FU Wei。吴翼为项目论文挂名最后一位，任清华大学交叉信息研究院助理教授，在知乎宣传项目，[如何评价开源训练框架 AReaL](https://www.zhihu.com/question/1890112252100703430)。推测 kdada 与 MEI Jun 为蚂蚁工作人员。

| 姓名 | 角色 | 单位 | 邮箱 |
| ------- | ------- | ------- | ------- |
| [吴翼](https://github.com/jxwuyi)  | 通讯作者  | 清华大学 上海期智研究院 边塞科技   | jxwuyi@gmail.com |
| [MEI Zhiyu](https://github.com/nuzant)  | 通讯作者 一作   | 清华大学 上海期智研究院   | meizy20@mails.tsinghua.edu.cn |
| [FU Wei](https://github.com/garrett4wade)  | 二作 最活跃 Commitor   | 清华大学 上海期智研究院   | \ |
| [kdada](https://github.com/kdada)  | 活跃开发者   | 杭州   | me@imkira.com |
| [MEI Jun](https://github.com/JacksonMei)  | 活跃开发者   | 蚂蚁集团(github) Momenta(LinkedIn 可能没更新)  | mjautoman@163.com |
| [XU Shusheng](https://github.com/xssstory)  | 活跃开发者   | 清华大学 | xssstory@gmail.com |


## 社区运作方式

该项目的代码全部开源，并托管在 Github 上，代码贡献者大家并无明确的分工界限，权限大致无差。项目 PR 相对较少，并没有太多项目核心外部人员参与合入。主要代码库仅有一个 [AReaL Github](https://github.com/inclusionAI/AReaL)。

文档：团队于 2025/04/27 发布项目[官方文档](https://deepwiki.com/inclusionAI/AReaL)，详细介绍了项目代码架构。

AReaL 团队目前计划以后的稳步发展以及进行稳定进行版本发布，未来的工作开展方向主要包括：

系统方面：支持强化学习中 coding problems，异步生成与强化学习训练，分布式训练优化，Expert Parallel 和 Zero-Bubble Pipeline 等

算法方面：完成 32 B 的模型 recipes，样本高效的多任务强化学习算法，等
