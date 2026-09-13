# Phase 02：标注交接与人工回放训练转换

**执行状态：代码和本批正式转换均已完成。** `humanM9_569` 于北京时间 2026-09-12 16:28:08 通过完整校验并发布，退出码 0。跨仓库目标与剩余 TODO 见 [Enactive 总览](/workspace/glx/projects/Enactive/data/developments/human-vln-implementation/README.md)。

## 目标

将已准备、已标注的人工回放转换为 Enactive VLN 使用的 LeRobot v2.1。四路视频第 i 帧对应第 i 行位姿，`timestamp=i/10`，state/action 是相对首帧的米制 xyz+xyzw；英文任务来自本次标注。

## 已完成

| 工作 | 完成结果 |
|---|---|
| 标注结果交接 | navigation-process 报告提供公开 annotation 合同；转换器直接核对 manifest/report/instruction，不再请求模型 |
| 训练转换入口 | `human_replay.py`、`scripts/human_replay.sh`、独立 source/converter 已实现，复用 Unreal/A* 坐标与 writer |
| 输出合同 | 四路 RGB、真实相机参数、N 帧/N 位姿、完整任务、来源 extras、modality、跨 episode 连续 index |
| 小样本与回归 | xNav 56 项通过；标注交接阶段 48 项通过、1 项跳过；真实 39 帧及多 episode 可被 LeRobot 加载 |
| 正式转换 | 619 条候选中输出 569 条，沿用 50 条标注质量排除；257,647 帧、10 FPS、约 7.16 小时、2,276 个视频；转换失败 0 |
| 完成判定 | 所有输出视频帧数/PTS/FPS、parquet 位姿/时间/任务、metadata/统计及 LeRobot 检查通过，正式目录发布，退出码 0 |

最终成功 UUID 集合与标注成功的 569 个 UUID 完全一致。正式运行使用 32 个 writer 和 32 个编码进程，转换报告记录耗时 2,878.55 秒。早期 2/16 路中断现场保留在本次运行目录，不属于正式数据集。

## TODO 与使用边界

本批训练格式转换没有剩余任务，无需再次执行。使用说明和新任务命令见 [转换器 README](../README.md)。

测试格式已从相同源回放导出，569 条的 UUID、帧数、指令和来源记录与训练一致；三个 episode 的首/中/末帧坐标转换抽样也通过。发布证据及后续 Windows 场景上传、UE 运行安排由 [Phase 03](/workspace/glx/worktrees/NavBench-humanM9-569/docs/human-replay-export/implementation/phase-03-navbench-export-plan.md) 维护，无需重做本批训练数据。训练混合比例、动作窗口、归一化统计和模型训练属于后续训练配置工作。

转换器当前没有整库断点续写功能。已有目标或同名 staging 会被拒绝覆盖；这是现有能力边界，本批已完成数据不需要补做恢复逻辑。

## 正式产物与证据

- 输出：[humanM9_569](/data/glx/Enactive/navigation/datasets/train/DataEngine/humanM9_569)。
- [最终转换报告](/data/glx/Enactive/navigation/datasets/train/DataEngine/humanM9_569/meta/human_replay_conversion_report.json)：`status=completed`、`num_successful=569`、`num_excluded=50`、`num_failed=0`、`total_frames=257647`。
- 元数据：输出目录下 `meta/info.json`、`episodes.jsonl`、`episodes_extras.jsonl`、`episodes_stats.jsonl`、`tasks.jsonl` 和 `modality.json`。
- 日志与退出码：`/data/glx/Enactive/navigation/staging/humanM9_569-conversion-20260912/train.log`、`train-exit-code.txt`。

## 历史批准与实现证据

以下记录描述最初“先实现、小样本验收”的阶段。其“尚未全量转换”“留到下一轮”等内容不代表当前状态；正式运行结果以上文为准。

<details>
<summary>展开原阶段计划、实现和小样本验证</summary>

## Approved Plan

2026-09-12：用户批准实现两个仓库内的标注交接和训练转换，并明确排除 Enactive 与 Benchmark 改动。

## Implementation Adjustments

1. 实施文档放在 xNav-data-preprocess 中，遵从本轮“不修改 Enactive 项目”的范围；原 Enactive 设计文档只读。
2. 标注报告新增公开的 annotation 合同，转换前用原有 report 命令刷新；转换器不依赖标注仓库 Python 包或 SQLite 内部表。此举不触发 API，不需要停止正在运行的标注任务。
3. 使用独立的 `utils/human_replay/source.py` 和 `converter.py`，继承 `UnrealEpisode` 的逐帧实现；不继承会全量载入 frames 或依赖地图的 collection。现有公共 writer、Unreal/A* 转换代码没有改动。
4. 发布前将本数据集的 `index` 规范化为跨 episode 连续的全局行号；保留每条独立的 frame_index、原位姿和 i/10 时间戳。该步骤只改新 staging 产物，不修改已有 A* 数据。
5. 现有 writer 有些 worker/编码异常会在子进程中记录后继续退出，因此成功判定额外检查实际文件、统计和 UUID 覆盖，不能只看 `creator.wait()` 返回。输入中途报错时关闭本次创建的 worker/encoder 及队列；不实现失败后自动追加或重试整个数据集。
6. 统计检查只要求 writer 实际拥有的 pose/action/RGB 统计，任务索引按映射校验；不要求为离散 annotation index 生成数值归一化统计。

## Implementation Record

### 代码与入口

- navigation-process：`outputs.generate_report` 仅对 human-replay 增加 `annotation.job_fingerprint/model/prompt_version/sampling_fps`；补充交接测试与 README。指令 JSON、模型请求、批处理调度保持当前行为。
- xNav：`human_replay.py` 提供必需路径、可选明确 UUID、worker 数；`scripts/human_replay.sh` 串联本地报告刷新和转换。新 source 核对标注与来源，逐条加载帧；新 converter 复用 writer 并负责 staging、严格产物验收、报告和发布。
- 新输出包含标准 v2.1 metadata/parquet/四路 RGB、pose VLN modality 和原始来源 extras。质量排除与未完成/错误分开记录；目标或 staging 已存在时拒绝覆盖。
- 本轮未提交、推送；实现仍在工作区。没有修改 Enactive 或 UE Benchmark 子模块，也未开始全量转换或训练。

### 测试证据

- 修改前：Unreal、A* 与准备器 39 项通过。
- 修改后 xNav：`test_human_replay_conversion.py` 加上述原测试，**56 passed**。覆盖来源/run/指纹冲突、缺标注与显式子集、质量过滤、短片和重复位姿、非均匀原时间、损坏索引/帧数/pose/K/外参/FPS；人工与 AStarEpisode 数值一致；1/2 worker 真实写入、LeRobotDataset 读取第二个 episode、全局 index、重复输出拒绝，以及编码失败即使 metadata 完整也不发布。
- navigation-process 全仓测试：**48 passed, 1 skipped**。保留旧 source 回归，新增报告与 instruction provenance 对应检查。
- Ruff、新脚本 bash 语法和 Git whitespace 检查通过；没有增加依赖。LeRobot PyAV 解码路径有现有 torchvision 弃用提示，不影响本次加载结果。

### 真实小样本验收

2026-09-12，仅选择 UUID `73eb06d6-b239-461f-b4ac-b7d69559f4f9`，由 `scripts/human_replay.sh` 刷新真实报告并执行完整转换。输出：

```text
/data/glx/Enactive/navigation/staging/human-replay-training-validation-20260912/shortest-vln
```

转换退出 0，1 条选中、1 条成功、0 排除、0 失败，共 39 行 pose 和四路各 39 帧 640×480 / 10 FPS 视频；时间戳为 0.0～3.8 秒，完整播放长度 3.9 秒。视频已完整解码核对帧数/PTS，元数据和任务映射通过 LeRobot 检查。原输入/标注未修改。

端到端命令用时 20.91 秒，`/usr/bin/time -v` 的 Maximum resident set size 为 1,101,392 KB；这只是短样本进程统计，不代表最长轨迹或多 worker 的整树峰值。日志与转换报告分别位于上层 `convert-shortest.log` 和数据集 `meta/human_replay_conversion_report.json`。

下一步仅是按选定范围执行正式转换；现有标注尚未完整结束时，默认全量入口会明确失败，不把部分标注发布成完整训练集。是否独立划分测试 UUID，以及 Enactive 配置和 Benchmark 导出仍留到下一轮。


</details>
