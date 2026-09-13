# 人工回放 → LeRobot VLN

`human_replay.py` 读取已准备的人工回放及 navigation-process 的标注报告，输出独立 LeRobot v2.1 数据集。实现及验证记录见 [阶段文档](implementation/phase-02-training-conversion-plan.md)。

## 本批目标与完成状态

本批目标 `DataEngine/humanM9_569` 已于北京时间 2026-09-12 16:28:08 完成：569 条、257,647 帧、10 FPS、2,276 个视频，完整校验通过，退出码 0。619 条候选中的 50 条沿用标注质量排除，转换失败 0。

[正式训练数据](/data/glx/Enactive/navigation/datasets/train/DataEngine/humanM9_569) 和 [最终报告](/data/glx/Enactive/navigation/datasets/train/DataEngine/humanM9_569/meta/human_replay_conversion_report.json) 已落盘，本批无需重跑。对应测试格式也已交付，双格式 UUID、帧数和指令对照通过；后续 Windows 场景上传与实际运行安排见 [Enactive 总览 TODO](/workspace/glx/projects/Enactive/data/developments/human-vln-implementation/README.md#todo补齐场景并验证运行)。

## 输入与结果交接

输入根目录包含 `manifest.jsonl` 和 `episodes/<UUID>/`。每条使用 `episode_meta.json`、`task_meta.json`、`replay.json`、`frames.jsonl`、四路 `rgb/{front,rear,left,right}.mp4` 以及本次生成的 `instruction.json`。

标注报告必须来自当前 navigation-process 的 `report` 命令，包含 `annotation` 合同。转换器按 UUID/revision/pipeline fingerprint 匹配条目，核对 run、job fingerprint、模型、v2 prompt、完整帧数和 10 FPS。只接受 `instruction_written` 或模型明确的 `filtered`；缺标注、API 失败、来源冲突会阻断所选范围的发布。原中文任务保存为溯源信息，不用于训练任务文本。

## 命令

本批正式转换采用以下命令；这是已完成任务的命令记录，目标已存在，再次执行会拒绝覆盖：

```bash
HUMAN_REPLAY_OUTPUT_ROOT=/data/glx/Enactive/navigation/datasets/train/DataEngine/humanM9_569 \
  bash scripts/human_replay.sh --num-workers 32
```

脚本会先刷新本地标注报告，不调用模型，然后转换全部准备清单。复用时通过 `HUMAN_REPLAY_OUTPUT_ROOT` 明确指定新的输出目录。

小批验收使用明确 UUID 和独立输出目录；可重复 `--episode-id`：

```bash
HUMAN_REPLAY_OUTPUT_ROOT=/path/to/new/sample-vln \
  bash scripts/human_replay.sh \
  --episode-id 73eb06d6-b239-461f-b4ac-b7d69559f4f9 \
  --num-workers 1
```

`HUMAN_REPLAY_WORK_ROOT`、`HUMAN_REPLAY_RUN_ID` 可选择其他准备目录及标注任务。已有输出与同名 `.staging` 都拒绝覆盖。转换不自动执行训练或测试集划分。

其他调用方可以直接传公开文件，转换器不依赖 navigation-process 的 Python 包：

```bash
.venv/bin/python human_replay.py \
  --input-root /path/to/prepared-replay \
  --annotation-report /path/to/annotation-job/report.json \
  --output-root /path/to/new/lerobot-vln \
  --num-workers 1
```

默认只使用 1 个 writer 和 1 个视频编码进程。增加 worker 前先用代表性样本测量资源：复用的 writer 会产生临时 PNG、重新编码 MP4，并在每条 episode 内计算图像统计，资源需求随最长轨迹和 worker 数增加。

## 输出合同

- 四路视频顺序和帧数不变，N 个干净帧对应 N 行 pose，不二次去重、不补插、不剪尾；原时间间隔不控制取帧，输出 timestamp=i/10。
- 复用 UnrealEpisode 的 UE cm/degree → 米制右手坐标 → 相对首帧变换，保存 `[tx,ty,tz,qx,qy,qz,qw]`；同帧 `action=observation.state`。训练动作窗口由后续 Enactive 配置拥有。
- 保留真实 K、固定外参和 scene/username。`meta/episodes_extras.jsonl` 记录 UUID、源 revision、回放处理指纹、标注 provenance、原采集时长/首末原时间戳、回放时长、原任务及源文件位置；完整原时间序列留在源 `frames.jsonl`。
- `meta/modality.json` 使用现有 pose VLN 的 state/action/video/annotation 映射，不声明缺失的地图或 depth。
- 基础 `frame_index` 从每条 episode 的 0 开始，`index` 在输出数据集内连续。旧 writer 的每 episode 局部 `index` 在新输出发布前规范化；不修改旧数据或公共 writer。
- 标准文件：`data/chunk-*/episode_*.parquet`、四路 `videos/chunk-*/video.*/episode_*.mp4`，以及 `meta/{info,modality}.json`、`tasks.jsonl`、`episodes.jsonl`、`episodes_stats.jsonl`、`episodes_extras.jsonl`。

## 完成与失败

先写 `<output-root>.staging`；完成后检查实际 UUID 覆盖、所有元数据/统计、parquet 位姿与任务、四路完整视频帧数/PTS/FPS，并用 LeRobot metadata loader 验证路径。通过后才将 staging 重命名为目标目录。

报告位于 `meta/human_replay_conversion_report.json`，列出所选 UUID、正常质量排除、失败、实际成功 UUID → episode_index 映射和帧数。异常退出时 staging 与报告保留，退出码非零，正式目标不会发布。报告中的 `num_submitted` 只表示进入 writer 队列，不能当作成功数。

当前不提供整库断点续写或覆盖开关。中断后先检查报告和已有文件，再决定修复或使用新的输出目录；不要直接追加到残缺目录。本批全量训练转换已完成；后续模型训练配置与 NavBench 测试交付由各自项目负责。
