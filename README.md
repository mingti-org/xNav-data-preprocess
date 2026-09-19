# xNav Data Preprocess

## 环境配置
本仓库使用[`uv`](https://docs.astral.sh/uv/getting-started)管理环境，安装好uv后使用如下命令配置环境

```bash
uv sync --all-groups
```

该命令会将环境装在项目目录下的`.venv`中，之后使用`uv run xxx.py`即可用该环境跑某个python程序。

## 概述
本仓库包含xNav对一些开源数据集的格式转换，详见[`docs`](./docs/)。

- [`rgb_pose_to_lerobot.py`](rgb_pose_to_lerobot.py): 给第一视角web video封装了一份代码，详见[`rgb_pose_example/README.md`](examples/rgb_pose_example/README.md)。
- [`lerobot_creator_example.py`](lerobot_creator_example.py): 教程示例代码。
- [`unreal.py`](unreal.py): 将 `3d-simu-ue` 录制出的 raw episode 转为按 scene 组织的 LeRobot v2.1 数据集；使用说明见脚本顶部注释。
- [`human_replay.py`](human_replay.py): 将完成 v2 英文标注的人工回放转为四视角 LeRobot v2.1 VLN 数据；[输入合同、命令及失败处理](docs/human-replay/README.md)。
- [`tracking.py`](tracking.py): 将 EVT-Collect raw/四视角导出按实测位姿转为 LeRobot v2.1；[位姿来源、直接输出目录与并行转换](docs/tracking-conversion.md)。

## Map2Nav VLN-CE replay

`map2nav_vlnce.py` 将单层 R2R/RxR replay 转成 Enactive 可读取的 LeRobot v2.1
pose-only 数据。一个物理轨迹中的每条指令会分别生成一个独立 episode，并在
`tasks.jsonl`、`episodes.jsonl` 和帧级 `task_index` 中保存真实指令。

转换 `rxr_guide` 时必须通过 `--rxr-annotations` 传入权威的 guide annotation。
转换器按 `episode_id` 校验文本、trajectory 和 scene，只保留 `en-US` 与 `en-IN`；
不会根据 replay 中的语言字段或字符集猜测语言。源 replay 的数据标识可以是
`rxr/guide` 或预先筛选英文后的 `rxr_en/guide`。

转换始终以成功的 `manifest.jsonl` 为输入。`errors.jsonl` 中已记录的 replay 失败项
不会阻断转换，其数量会写入 `conversion_report.json` 的
`source_recorded_errors` 字段；失败目录不会被扫描或转换。

```bash
.venv/bin/python map2nav_vlnce.py \
  --input-root /data/glx/indoor_data/map2nav/r2r_replay_4_view_2048 \
  --output-root /data/glx/indoor_data/map2nav/processed_v2/r2r \
  --dataset-name r2r \
  --split train \
  --flat-output \
  --num-workers 32

.venv/bin/python map2nav_vlnce.py \
  --input-root /data/glx/indoor_data/map2nav/rxr_replay_guide_4_view_2048 \
  --output-root /data/glx/indoor_data/map2nav/processed_v2/rxr \
  --dataset-name rxr_guide \
  --split train \
  --rxr-annotations /data/glx/indoor_data/habitat/data/vln_ce/raw_data/rxr/train/train_guide.json.gz \
  --flat-output \
  --num-workers 32
```

固定的 train-only wrapper 使用 32 个 worker，一次执行会先转换 R2R，成功后再转换 RxR：

```bash
bash scripts/map2nav_vlnce.sh
```

wrapper 直接写入 `processed_v2/r2r` 与 `processed_v2/rxr`，不会增加中间的
`train/` 目录。

默认拒绝写入已存在的 split。中断后只可在 conversion context 完全一致时使用
`--resume`；`--overwrite` 会删除并重建所选 split，使用前应单独确认目标路径。

### ScaleVLN replay

同一入口支持 `--dataset-name scalevln`。`--input-root` 可以指向一个含有
`train/manifest.jsonl` 的单 shard，也可以指向含有 `shard_*/train/manifest.jsonl`
的总目录。总目录按 shard 名字排序、按各自 manifest 顺序转换成一份数据集，
episode、task 和全局帧索引统一连续编号；重复的指令 episode ID 会在写出前报错。
同一输入目录同时包含直接 split 和 shard 时会报错，避免遗漏或重复转换。

ScaleVLN 复用 R2R/RxR 的位姿、动作、四视角视频和 Parquet 输出合同，地图只要求
`graph.png` 与对应的轨迹覆盖图。投影由每条轨迹所属 shard 的 graph metadata
重建，并继续与记录的像素坐标核对。R2R/RxR 仍要求原来的六张地图，RxR 仍按权威
标注只保留英文指令。

ScaleVLN 的 `floor_level_id=-1` 表示没有楼层标注，不作为单层证明。该数据集不做
楼层筛选，报告写入 `floor_filter="not_applied"`、
`floor_filter_reason="source_has_no_floor_metadata"` 和 `eligible_source_episodes`，
不输出单层/多层筛选计数。Python 接口的 `skip_preflight=True` 使用相同规则，
实际转换仍校验所选轨迹的 steps、位姿、动作和地图投影。

只转换成功 manifest 中的记录，各 shard 的 `errors.jsonl` 数量汇总为
`source_recorded_errors`。`source_episode_dir` 在多 shard 输入时包含
`shard_*/train/` 前缀；resume 会检查 shard 列表和每条输出的来源，不能把另一个
shard 的单独转换追加到已有输出中。

```bash
python map2nav_vlnce.py \
  --input-root /data1/glx/Enactive/datasets/train/scalevln/raw/0918_scalevln_40k \
  --output-root /path/to/processed/scalevln \
  --dataset-name scalevln \
  --split train \
  --flat-output \
  --num-workers 32
```

这个转换器直接写 Parquet 并复制视频，不调用 LeRobot SDK。所用 Python 环境需要
NumPy、SciPy、Pillow、PyArrow 和 tqdm；定向测试另外需要 pytest、pandas 和 OpenCV。
仓库中使用 LeRobot SDK 的其他转换入口仍需要各自的依赖。
