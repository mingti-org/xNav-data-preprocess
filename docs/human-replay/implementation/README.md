# 人工回放训练转换实施

## 目标与状态

将 navigation-process 的公开标注结果转换为 Enactive-PI / LeRobot v2.1 训练格式。**实现、回归、小样本和本批正式转换均已完成**；详细证据见 [Phase 02](phase-02-training-conversion-plan.md)。

| 交付项 | 状态 |
|---|---|
| 人工 reader、转换入口和验证 | 已实现；56 项测试通过，真实 39 帧样本可加载 |
| `DataEngine/humanM9_569` | 已发布；569 条 / 257,647 帧 / 2,276 个视频，2026-09-12 北京时间 16:28:08 完成，退出码 0 |
| 本阶段 TODO | 无；NavBench 测试格式和双格式对照也已完成，后续场景资产与运行待办见总览 |

## 基线与导航

- 训练转换代码基线：xNav-data-preprocess `1b559e8c0ff047657dfe14bd241d16db4915592a`；标注交接基于 navigation-process `fd7f174948f9abd1ff6ad37a90168607719b232c` 及该工作区的人工标注实现。
- [冻结设计](/workspace/glx/projects/Enactive/data/developments/2026-09-12-human-vln-replay-annotation-conversion-plan.md) 第 5、7.2 节，SHA256 `64e0fd11e904961db94564bc400f85fe97d91c8edfa60aae2694ce98bea21732`，作为历史基线保留。
- [转换器使用说明](../README.md) 提供输入/输出合同及新任务命令。
- [Enactive 总览](/workspace/glx/projects/Enactive/data/developments/human-vln-implementation/README.md) 维护当前两种格式的目标、已完成和 TODO；[NavBench Phase 03](/workspace/glx/worktrees/NavBench-humanM9-569/docs/human-replay-export/implementation/phase-03-navbench-export-plan.md) 维护场景待办。

实现文件仍在 xNav 工作区；本任务没有推送或启动模型训练。
