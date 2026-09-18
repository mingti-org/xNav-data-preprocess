# HM3D OVON raw → LeRobot v2.1 转换方案

状态：Draft，等待独立方案审核

源数据：`/data/glx/Enactive/navigation/datasets/train/ovon/raw/ovon_0918_40k`

代码仓库：`/workspace/glx/projects/xNav-data-preprocess`

## 1. 目标与非目标

本方案把 OVON HM3D 仿真采集结果转换为 Enactive 使用的 LeRobot v2.1 数据集，保留四视角视频、逐帧位姿、离散动作和 ObjectNav 任务文本，使输出可由现有 `habitat_discrete_pose` 数据适配器读取。

本方案不修改 raw 源数据，不重跑 Habitat/UE 采集，不处理 `errors.jsonl` 中失败轨迹，不改变官方 OVON 任务语义，不把 `back` 作为 LeRobot 视角名，也不在本任务中启动全量转换或训练。

## 2. 已确认的输入合同

输入下有 `shard_0/train` 和 `shard_1/train`。每个 shard 包含成功 `manifest.jsonl`、失败 `errors.jsonl`、`run_manifest.json`，以及 `episodes/train/<episode_dir>/`。

成功 episode 目录至少包含：

```text
episode.json
steps.jsonl
trajectory.npz
front.mp4
back.mp4
left.mp4
right.mp4
```

采集配置为 640×480、10 FPS、四视角、HFOV 120°、前进 0.25 m、转角 15°、最大 500 步。

当前审计基线：成功 manifest 39,281 条、失败记录 719 条、成功轨迹总帧数 1,800,744。`episode_id` 在 shard 间大量重复，不能作为唯一主键；唯一来源键使用：

```text
source_episode_key = <shard_name>/<episode_dir_name>
```

例如 `shard_0/hm3d_1S7LAXRdDqK_traj_10519` 与 `shard_1/hm3d_6imZUJGRUq4_traj_10519` 必须作为两个 episode。

## 3. 输出合同

输出目录由命令行显式指定，默认建议：

```text
/data/glx/Enactive/navigation/datasets/train/ovon/processed/ovon_0918_40k
```

写入 `<output>.staging`，完成全部校验后才发布为目标目录；目标已存在时默认拒绝，只有显式 `--overwrite` 才允许由调用者确认后重建。

LeRobot video feature 必须使用 canonical 名称：

```text
video.front  ← raw front.mp4
video.rear   ← raw back.mp4
video.left   ← raw left.mp4
video.right  ← raw right.mp4
```

每个成功源 episode 生成一个 LeRobot episode。任务文本优先取 `episode.json.instructions[0].instruction`；缺失时才按 `object_category` 生成 `Navigate to the {object_category}`，并在报告中标记 fallback。不得从目录名猜类别。

逐帧字段：

```text
observation.state  = steps.jsonl 的 position[x,y,z] + rotation[x,y,z,w]
action             = steps.jsonl 的 discrete_action_to_next_id
action_text        = steps.jsonl 的 discrete_action_to_next
timestamp          = step_index / video_fps
```

实际输入的最后一行已经包含终止动作：样本审计确认 `discrete_action_to_next == "STOP"`、`discrete_action_to_next_id == 0`，且该行仍对应最后一个视频帧。因此不追加帧、不覆盖最后一帧；固定动作合同为 `action_id=0`、dtype `int64`、shape `[1]`、名称 `STOP`。已观察到的非终止动作 ID 为 `1=forward`、`2=turn_left`、`3=turn_right`；实现必须全量核对集合，未知 ID/名称阻断 episode。原始数值字段、source key、scene、category、navigation metrics 写入 episode extras/报告。

`steps.jsonl` 是逐帧权威输入：`position` 为米制 `[x,y,z]`，`rotation` 已按 `[qx,qy,qz,qw]` 存储，`step_index` 与 `video_frame_index` 从 0 连续递增。`trajectory.npz` 必须存在且包含 `positions[N,3]`、`rotations[N,4]`、`discrete_action_to_next_ids[N]`、`video_frame_indices[N]` 等数组；实现逐元素与 JSONL 比较，任何 shape、数值或索引冲突阻断 episode。NPZ 不替代 JSONL，二者不一致不得静默选择一方。

## 4. 实现方案

新增 `ovon.py` 作为唯一入口，复用 `utils.lerobot.lerobot_creater.LeRobotCreator`、现有视频检查工具和 metadata loader；不复制另一套 LeRobot writer。

### 4.1 CLI

```text
python ovon.py \
  --input-root /data/glx/Enactive/navigation/datasets/train/ovon/raw/ovon_0918_40k \
  --output-root /data/glx/Enactive/navigation/datasets/train/ovon/processed/ovon_0918_40k \
--num-workers 1
```

参数：`--input-root`、`--output-root`、`--num-workers`、`--overwrite`、`--resume`、`--audit-only`。首版强制 `--num-workers=1`：按 source key 排序后串行提交，利用现有 `LeRobotCreator` 动态分配保证 episode index 稳定；不扩展 writer 的动态 index API。其它值直接报错。默认不覆盖、不恢复不一致上下文；`--resume` 必须校验输入 fingerprint、配置 fingerprint 和输出 schema 完全一致。

### 4.2 输入发现与 source manifest

`discover_sources()` 按 shard 排序读取成功 manifest，使用字段 `episode_dir` 相对于 `<shard>/train` 拼接；`resolve()` 后必须位于该 shard 的 `train` 根内，拒绝绝对路径、`..` 逃逸和路径不存在。构造不可变 `SourceEpisode`。同一 manifest 重复行、同一 `source_episode_key` 重复均为阻断错误，不做集合去重。读取并校验 `episode.json`，从中取得 instruction、category、scene 和导航指标。

`validate_source_episode()` 检查目录、四个 MP4、JSONL、NPZ、episode 字段、步数和视频 metadata。所有失败写入 `source_errors.jsonl`；已记录的采集失败单独统计为 `source_recorded_errors`，不混入转换错误。

### 4.3 帧读取与动作映射

按 `steps.jsonl` 顺序读取每行，验证 `step_index` 连续、`video_frame_index` 与行号一致、`num_steps` 一致。使用 position 和 quaternion 写入状态；使用 `discrete_action_to_next_id` 写入 action，并保留动作名称。动作字典从数据中收集并与预期 `stop/forward/turn_left/turn_right` 显式核对；未知动作阻断该 episode，不静默映射。

仓库 `LeRobotCreator` 的 worker 接收逐帧 image 数组并通过 `encode_video_frames` 在 staging 的 `videos/chunk-*/video.<view>/` 生成 LeRobot 视频；它不接受 raw MP4 路径作为 frame 值。因此实现必须按 video frame index 解码 raw MP4，逐帧提交四路 RGB，输出由 writer 按固定 `h264/yuv420p/10fps` 编码，不能直接把 MP4 路径写入 parquet。raw `back` 只作为输入文件名，输出 feature 始终是 `video.rear`。每个视角验证帧数、宽高、FPS 和可解码性；任一路编码/解码失败则该次发布失败。

### 4.4 LeRobot 写入

创建固定 features：四路 video、7 维 state、离散 action、timestamp/frame index，以及由 `LeRobotCreator` 自动管理的 episode/task 索引。转换前先扫描全部可转换 source episode，收集 task 文本并按字典序唯一化；主进程按该顺序调用 `creator.add_task(task)`，建立 `task_text → task_index` 表，worker 只使用已注册整数，不隐式新增 task。每帧注册同一个 task 文本；每个 source episode 结束后调用 episode flush。episode index 按 source key 字典序串行提交，由现有 metadata service 动态分配；两者及 source key 均写入 metadata/extras，排序规则、输入 fingerprint 和 schema fingerprint 共同构成 resume context。metadata extras 记录 `source_shard`、`source_episode_id`、`source_trajectory_id`、`source_episode_dir_name`、`scene_key`、`object_category` 和 navigation metrics。

### 4.5 报告与发布

报告计数必须满足闭合公式：`source_recorded_errors = 实际 errors.jsonl 总数`，`converted_episodes + conversion_errors + skipped_episodes = source_success_manifest_rows`；首版 `skipped_episodes` 恒为 0，不提供可恢复跳过策略。任何源校验或转换错误都计入 `conversion_errors` 并阻断发布；`converted_episodes` 仅在该 episode 的 flush、metadata 和视频编码均成功后计数。仅 `--audit-only` 可以输出不可发布报告。每条错误必须带 source key 和阶段。episode 已部分写入后，保留 staging 并标记 `publishable=false`，不得把残缺目录发布为正式输出。staging 中写入 `conversion_report.json`、`validation_report.json`、`source_manifest.jsonl`、`errors.jsonl`。全量验证成功后，使用原子 rename 发布。

## 5. 代码结构与测试

计划新增：

```text
ovon.py
test/test_ovon_conversion.py
```

测试覆盖：

- raw `back` 映射为 `video.rear`
- 跨 shard 相同 `episode_id` 不冲突
- 缺文件、坏 JSON、步数不匹配、未知动作被报告并阻断对应 episode
- instruction 优先级和 category fallback
- 最后一帧 stop/no-op 合同
- 四视角帧数和 metadata 校验
- staging、重复输出拒绝、resume context 校验
- LeRobot metadata loader 能读取至少两个不同 shard 的 episode
- fixture 必须包含两个 shard、相同数字 `episode_id`、真实 JSONL/NPZ 字段、`STOP=0` 最后一帧和四路短视频；验证 source key 集合、总帧数、feature shape/dtype、`video.rear` 存在且 `video.back` 不存在
- 验证输出 episode source key 集合与可转换 source manifest 完全相等，且所有 episode 行数及总帧数闭合

fixture 固定放在 `test/fixtures/ovon_minimal/`，由测试内的 Python/OpenCV helper 在临时目录生成 2 帧 2×2 的 H.264 MP4、JSONL 和 NPZ；测试不访问 `/data/glx/...`，不依赖 GPU。fixture 包含两个 shard、重复数字 episode ID、不同 scene、完整 `episode.json`/`steps.jsonl`/NPZ 和 `STOP=0`。测试在 `--num-workers=1` 下运行，并在 finally 中调用 creator 的 `wait()`/进程清理；若视频编码依赖不可用，测试明确失败而不是跳过。

静态验证包括：`.venv/bin/python -m compileall ovon.py utils test`、`pytest -q test/test_ovon_conversion.py`、ruff（若项目配置启用）、shell/JSON/YAML 语法检查，以及输出 schema 的离线 metadata 验证。静态验证不等于全量视频解码或训练正确性；全量转换和训练属于后续明确授权范围。

## 6. 实现顺序

1. 完成 source discovery、复合 key 和 audit report。
2. 完成 `steps.jsonl`/`episode.json` schema 与 action/state 映射。
3. 接入 `LeRobotCreator`，实现 `back → video.rear`。
4. 加入 staging、报告、resume/overwrite 保护。
5. 编写单元测试和小型临时 fixture。
6. 运行静态验证并记录结果。

## 7. 停止条件

独立方案 reviewer 需确认上述输入/输出合同、`rear` 命名、复合 episode key、最后一帧动作语义、错误处理和不启动全量转换均无问题。方案审核通过后才开始代码实现；实现后再由独立 reviewer 审核代码。代码审核和静态验证完成后停止，不自动发布数据、不启动训练。

## 8. 审核与实现记录

本节仅追加记录，不改写前述合同。

### Approved Plan

第一轮独立审核：`CHANGES_REQUIRED`（F1–F7）。第二轮复审补充 F8–F11：首版单 worker 串行稳定 index、预扫描后按字典序调用 `add_task()`、固定 fixture 路径和无真实数据测试、`skipped_episodes=0` 且任何转换错误阻断发布。待复审。

### Review Finding Assessments and Dispositions

待填写。

### Implementation Record

待填写。

### Publication Record

本任务不包含数据发布或远程 push；待填写。
