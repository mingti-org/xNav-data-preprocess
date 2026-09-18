# OVON converter code review

结论：**CHANGES_REQUIRED**。当前 `ovon.py` 不能按 `ovon_conversion_plan.md` 的合同可靠地产生可发布数据；修复后再做静态验证。

## Findings

### F1 (阻断): `LeRobotCreator` 不会读取 callable iterator 的 metadata
`convert()` 给 `episode_iter.metadata` 赋值后，把 callable 传给 `submit_episode()`。但 `worker_service()` 在 `callable(item)` 分支只执行 `iterator = item()`，只有非 callable 分支才检查 `hasattr(item, "metadata")`。因此 source key、scene、category 等 extras 永远不会写入 `episodes_extras.jsonl`，违反 provenance 合同。应改变传递方式（传带 metadata 属性的可迭代对象而非函数，或修正 writer API/worker 使 callable 分支也复制 metadata），并加测试确认 extras。

### F2 (阻断): 源校验错误直接 abort，未生成约定的错误报告
`discover()` 将错误收集到 `source_errors`，但 `convert()` 遇到任何错误直接 `raise ValueError`。不会写 staging `errors.jsonl`/`conversion_report.json`，也不满足“每条错误带 source key 和阶段、保留 staging、publishable=false”的合同。应统一错误 ledger，阻断发布但持久化报告。

### F3 (阻断): 任务注册方案与实现不一致/缺少可验证 task index
代码先 `creator.add_task()`，但每帧仍通过文本传入，worker 又调用 `meta.add_task(t)`；虽然 metadata 层可能去重，仍绕过方案中“预注册、worker 只使用已注册整数”的约束，且没有保存或验证返回的 task index 映射。应显式建立映射并验证输出 `tasks.jsonl` 与 frame `task_index` 闭合。

### F4 (高): 输出 features 缺少方案要求的 `action_text`，且 `timestamp/frame_index` 依赖 writer 隐式生成
方案要求保留 `action_text`；代码没有该 feature/逐帧字段。`timestamp`、`frame_index` 是 creator 隐式加入，不在 features/schema 中显式定义，需确认下游合同并在验证报告中检查。若要保留动作名称，应新增正确 dtype/shape 的 feature 或 extras 设计。

### F5 (高): 视频合同验证不完整且内存开销失控
代码把每个 episode 四路视频全部 `list(_frames(...))` 后才提交，39k episode 会造成单 episode 的巨大内存峰值；同时只比较解码帧数，未检查 640x480、FPS、编码可读性/metadata。应逐帧同步迭代（或受控缓冲）并验证尺寸/FPS，确保异常能进入错误 ledger。

### F6 (高): manifest/episode 元数据一致性检查不足
`validate_episode()` 未核对 manifest 的 episode/scene/trajectory/category（若存在）与 `episode.json`，未验证 instruction 非空/格式，未检查 video frame index 与 MP4 元数据；`episode.json` 字段缺失会在后续 `sources` 访问处抛出裸异常。应将这些检查归入 source validation 并带 source key。

### F7 (中): source fingerprint 不完整，resume/配置合同未实现
代码 fingerprint 仅哈希 manifest 文件，CLI 虽声明 `--resume` 但未实现。没有 schema/config fingerprint，也没有 resume context 校验。应删除未实现参数或实现完整保护；至少不能宣称满足计划。

### F8 (中): overwrite/staging 具有破坏性且未覆盖异常清理语义
`overwrite=True` 直接 `shutil.rmtree(output_root)`，且会删除既有正式输出；计划要求显式确认虽有 flag，但实现未做更细粒度保护。转换中异常不会调用 `creator.wait()` 清理子进程，可能遗留 worker/encoder 和 staging。

### F9 (中): `--audit-only` 输出仅 stdout，不生成计划规定的审计清单
计划要求 source manifest/audit/error 文件；当前 audit-only 只打印 sources/errors，无法作为后续转换输入或独立审计证据。

### F10 (中): features/data 类型兼容性未验证
`observation.state` 使用 Python list 拼接后转 float32 尚可；视频 shape/names 写法需由当前 `LeRobotMetadata`/datasets 实际验证。`action` 使用 int64 `[1]` 但没有静态或 fixture 测试确认 parquet schema、stats、loader 能接受该类型。代码也未执行输出 metadata 闭合检查。

## Review verdict

在修复 F1–F5 后，补齐 source/error/report ledger、任务索引闭合、视频 metadata/内存处理和最小 fixture 测试，再进行第二次独立 code review。当前不应启动真实 40k 转换，也不应发布 staging。

## 复审（修订版）

结论：**CHANGES_REQUIRED（仍不可开始真实转换）**。

检查当前 `ovon.py` 与 `LeRobotCreator` 实现后，上一轮 F1--F10 的关键合同问题仍未全部修复：

### 阻断问题

1. **F1 仍未修复：metadata 传递没有可验证闭合。** `EpisodeIterator` 带有 `metadata`，但提交到 multiprocessing worker 后，metadata 是否被复制完全取决于 creator 的 callable/iterator 分支；当前没有 fixture 验证 `episodes_extras.jsonl` 包含 source key。更严重的是 iterator 内部异常发生在 worker，不能进入转换错误 ledger。

2. **F2/F9 仍未修复：错误和审计报告没有落盘。** `convert()` 遇到 `source_errors` 直接 `raise ValueError`，不会生成 staging `errors.jsonl`/`conversion_report.json`；`--audit-only` 只打印 stdout，不能作为后续转换输入或审计证据。

3. **F3 仍未修复：任务索引不闭合。** `creator.add_task(task)` 的返回整数被丢弃，worker 的 `WorkerEpisodeBuilder.finalize()` 又按帧文本隐式调用 `meta.add_task()`；没有固定映射，也没有验证 `tasks.jsonl`、parquet `task_index` 与 episode metadata 的闭合关系。

4. **转换异常无法安全收尾。** `creator.wait()` 不在 `try/finally` 中；worker/encoder 异常时不会生成 failure ledger，可能遗留子进程和不完整 staging。

### 高风险未解决问题

- `action_text` 使用 `dtype=string, shape=[1]`，但没有最小 fixture 验证 LeRobot v2.1 的 parquet、stats、loader 均可接受。
- 转换阶段没有检查输出 episode 长度、feature 集合、task index 和 source extras。
- 视频读取已逐帧 streaming，内存问题改善；但没有把每路实际 frame count 与 steps/episode metadata 严格闭合并写入报告。
- `steps` 为空时 `steps[-1]` 抛出裸异常；manifest 的 category/trajectory 等字段也未完整一致性验证。
- fingerprint 仅哈希 manifest，不含配置/schema/代码版本；`--audit-only` 仍非文件化审计输出；`--overwrite` 直接递归删除目标，缺少产物保护。

### 必须补齐后才可通过复审

1. 增加 metadata fixture，确认 `episodes_extras.jsonl` 每条包含复合 source key，并明确 iterator 传递路径。
2. 统一 source/conversion error ledger：错误写入 staging `errors.jsonl`，生成含 `publishable=false` 的 `conversion_report.json`，并保证异常清理 worker/encoder。
3. 固定 task 文本到整数 `task_index` 映射，禁止 worker 隐式注册；验证 tasks/frame/episode 三方闭合。
4. 用最小 fixture 验证 `action_text`、视频 features 的 schema、parquet 和 loader。
5. 审计模式生成 source manifest/audit/errors 文件，补足视频 frame count 一致性检查。

复审 verdict 保持 **CHANGES_REQUIRED**；不应启动 40k 全量转换或发布 staging。

## 独立复审（本轮，2026-09-18）

结论：**CHANGES_REQUIRED**。

本轮按 F1--F10 逐项复核当前 `ovon.py`，未发现足以改变上一轮结论的修复。关键证据如下：

- **F1/F2/F3 仍为阻断项。** `EpisodeIterator.metadata` 仍依赖 `LeRobotCreator` 的 worker 分支，未有 fixture 或输出检查证明 `episodes_extras.jsonl` 写入 `source_episode_key`；source validation 一旦出现错误仍只在内存收集后直接抛出，正常转换异常也没有统一 failure ledger；task 文本仍由 worker 隐式 `add_task`，`add_task()` 返回值未保存，未校验 `tasks.jsonl` 与 parquet `task_index` 闭合。
- **F4/F10 未闭合。** `action_text` 仍声明为 `string/[1]`，没有最小 LeRobot parquet/loader 验证；转换结束只写计数报告，不检查输出 features、episode lengths、task indices 或 extras。
- **F5 部分改善但合同未完成。** `EpisodeIterator` 已逐帧读取视频，避免了整集 `list()` 的峰值；但未统计并核对每路实际解码帧数，也未将 worker 内的视频异常写入错误报告。
- **F6/F9 仍未完成。** 空 steps 会在 `steps[-1]` 处产生未归档异常；manifest 与 episode 的 category/instruction/trajectory 完整一致性仍不足；`--audit-only` 只写简化的 `source_manifest.jsonl`/`errors.jsonl`/报告，没有计划要求的完整 audit 清单与 frame/video 闭合证据。
- **F7/F8 仍未完成。** fingerprint 只覆盖 manifest，未覆盖 schema/config/code；异常清理不在可靠的 `finally` 合同内，`--overwrite` 仍直接删除既有输出。

因此当前代码不能开始 40k 全量转换，也不能发布 staging。必须先补齐：可验证的 extras 传递 fixture、统一可落盘 error ledger 与异常清理、显式 task-index 映射及闭合校验、action/video schema 最小 fixture、完整 audit/frame-count 报告；完成后再提交下一次独立复审。

## 独立最终复审（2026-09-18）

结论：**CHANGES_REQUIRED**。本轮只读检查了当前 `ovon.py`、现有 `LeRobotCreator`/metadata writer、方案合同和上一轮 F1--F10；静态编译通过，但代码仍不能启动 40k 转换或发布 staging。

### 已确认的改进

- `EpisodeIterator` 以可迭代对象（而非 callable）提交，当前 worker 的非-callable 分支会复制 `metadata`；这是 F1 的方向性修复，但没有 fixture/真实最小输出证明 extras 已写入且字段完整。
- 源校验错误路径会尝试保留 `<output>.staging/errors.jsonl` 和 `conversion_report.json`。
- 四路 raw 视角映射保持 `back -> video.rear`，读取采用逐帧 streaming；输出任务文本预扫描并调用 `add_task()`。
- `.venv/bin/python -m py_compile ovon.py` 与 `git diff --check` 通过。

### 仍然阻断发布的问题

1. **依赖的 metadata writer 会重复写 episode。** `utils/lerobot/lerobot_creater.py:metadata_service()` 在 `CMD_APPEND_EPISODE` 分支连续调用两次 `meta.append_episode(args)`。因此每个 worker episode 至少产生两条 `episodes.jsonl` 记录；`_validate_output()` 会失败，但该失败发生在 `convert()` 的 `try/except` 之外，不会写入 `conversion_report.json`/`errors.jsonl`，也不会安全收尾。必须修复或绕开该 writer 缺陷，并用最小 fixture 验证 episode 行数。

2. **worker/encoder 异常仍无法进入统一错误账本。** `worker_service()` 和 `video_encoder_service()` 捕获异常后仅 `logging`，继续处理队列；主进程看不到 source key、阶段或异常。`creator.wait()` 也没有可靠的 `finally` 生命周期保护。视频编码失败可能被吞掉，转换仍写出 `publishable=true`，或稍后由 `_validate_output()` 裸抛异常。需要 worker-to-main error channel、每条错误的 source key/stage、统一失败报告及进程清理。

3. **任务索引只做 metadata map 检查，没有数据闭合。** worker 仍在 `WorkerEpisodeBuilder.finalize()` 内按帧文本再次调用 `meta.add_task()`，不是“只使用预注册整数”的合同；`_validate_output()` 只比较 `tasks.jsonl` 与主进程 map，没有读取每个 parquet 的 `task_index`，也没有核对 episode `tasks`、extras `episode_index` 和 frame task index 三方一致性。应显式传递固定整数或验证 writer 生成值。

4. **输出验证不足。** `_validate_output()` 只检查文件存在、任务字典、episode/extras 行数、source key 集合和总 metadata length；没有验证：
   - parquet 实际行数/episode_index 与 `episodes.jsonl` 对齐；
   - `info.json` 的 `total_episodes/total_frames/total_videos/features`；
   - 四路输出视频存在、可解码、实际帧数/尺寸/FPS；
   - extras 的 `source_shard`、`source_episode_id`、`source_trajectory_id`、scene/category、navigation metrics；
   - `action_text`/video feature 的 parquet schema 和 loader 兼容性。
   当前没有方案要求的最小 fixture 测试文件可供证明。

5. **输入审计和错误落盘仍不完整。** `discover()` 在 `manifest['episode_dir_name']` 缺失、重复 source key、manifest JSONL 解析失败等情况下会在统一 `try` 之外直接抛错；这些错误没有 source key/stage ledger。`validate_episode()` 未核对 manifest 与 episode 的 category/instruction/trajectory 全部字段，也没有解码并计数每路 raw 视频帧。源校验失败路径将 `source_recorded_errors` 固定写成 0，且转换路径没有写方案要求的 `source_manifest.jsonl`、`validation_report.json` 或完整 audit 清单。

6. **空轨迹/部分写入的失败语义未闭合。** 空 `steps` 虽最终会触发校验，但异常来源和 source key 不会进入 ledger；iterator 内解码提前结束时可能已分配 episode/写入临时图片。`_validate_output()` 或 task/metadata 校验失败时没有 `finally` 清理/停止 creator，也没有生成 `publishable=false` 报告。

7. **fingerprint 与方案上下文不一致。** `_fingerprint()` 只哈希两个 manifest 文件，未覆盖 schema、配置、代码版本；当前 CLI 也没有实现方案声明的 `--resume` 上下文校验。若首版不实现 resume，应从方案/报告中明确删除该承诺，并至少将 schema/config fingerprint 写入报告。

8. **覆盖保护和测试缺失。** `--overwrite` 直接递归删除目标目录，没有校验目标是否为本工具产物；没有 `test/test_ovon_conversion.py` 或 fixture 来验证 `back -> rear`、跨 shard 复合 key、extras、task index、视频/Parquet schema 和 failure ledger。

### 复审结论

当前仅静态语法验证通过；尚未达到 reviewer 要求的“实现完成后独立审核通过”。在修复 writer 重复 episode、统一 worker/encoder error ledger 与清理、显式 task-index/Parquet 闭合、完整 source/output audit、fingerprint/overwrite 合同，并补齐最小 fixture（含 `action_text`、四路视频和 extras）后，才可再次复审。不得启动 `/data/glx/Enactive/navigation/datasets/train/ovon/raw/ovon_0918_40k` 的真实转换。
