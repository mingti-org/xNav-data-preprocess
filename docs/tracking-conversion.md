# Tracking 四视角并行转换

`tracking.py --processed-root` 接收 EVT-Collect 的四视角 Stage-4 导出，输出一个 `train/` LeRobot v2.1 数据根。使用原有 `--workers` 参数同时控制 episode 输入校验与视频转换的并发数。

## 执行流程

1. 主线程读取全局 manifest、每条 episode 的 source manifest、相机元数据和 JSONL 首条非空行。根据声明的帧数和首行指令，按既有规则确定 episode、task、全局 frame 编号。此阶段不枚举或打开图片，也不解析完整 JSONL。
2. 每个 worker 读取本 episode 的完整 JSONL，核对实际帧数、指令、动作有效性、episode 身份、帧序号、时间戳和四视角路径。相同相机引用在该 worker 内只解析一次。图像路径仍检查符号链接解析后的根目录边界、episode/view 身份、普通文件类型和重复引用。
3. worker 按原顺序读取每张图片，在用于编码的同一次 `Image.open()` 中检查尺寸并解码 RGB，同时计算统计量。保留 x264 参数、10 FPS、四视角映射和原生视频完整解码校验。
4. 每条 episode 的 Parquet 和四路视频全部通过检查后，才计入完成进度。汇总按预先固定的编号生成元数据，不依赖 worker 完成顺序；原生数据集校验成功后原子发布最终目录。

异常输入仍使整次转换失败。错误发生时取消尚未执行的任务，等待正在执行的 worker 退出；中间产物保留在原生 staging 中。即使使用原有 `--overwrite`，转换失败也不会替换已有输出。

## 为什么会更快

旧实现先在主线程检查所有 JSONL 行及所有图片，再启动编码池；编码时又打开相同图片。现在逐帧校验随 episode 并行执行，尺寸校验复用编码读取，从每张图片两次打开减少为一次。

以 371100 个时间帧、四视角输入为例，取消了 1484400 次额外图片打开。同一 episode 的相机路径不再逐帧重复解析。主线程也不再保留全库 JSONL 行和图像路径，只保留轻量计划与相机元数据；完整逐帧数据由当前运行的 worker 持有。

主线程整理元数据仍是串行步骤，图像路径检查及编码仍有 Python、CPU 和共享存储开销，因此 worker 数增加不保证线性提速。没有新增“跳过校验”开关，也没有扩大成功数据选择范围。

## 进度日志

- `Indexed metadata for N/M episodes`：完成轻量清单整理，尚未表示已生成视频。
- `Converted N/M episodes, F/T frames, ... frames/s, conversion ETA ...`：已完成 episode 的校验、编码和视频验证。ETA 按本轮累计帧吞吐估算，早期和尾部可能波动。
- `Validating final dataset metadata`：视频阶段完成，进入最终元数据检查。前一条 ETA 不包含本阶段；最终退出与正式目录发布才代表完成。

## 使用方式

```bash
.venv/bin/python tracking.py \
  --processed-root /absolute/path/to/stage4 \
  --output-dir /absolute/path/to/lerobot \
  --work-dir /absolute/path/on/same/filesystem/work \
  --workers 64
```

输出与 work 需要位于同一文件系统，以支持现有的原子目录重命名。EVT-Collect 现有 `scripts/convert_tracking.py` 已传入上述参数，合并后可直接使用，不需要修改采集项目入口。

## 本轮实现与验证

2026-09-14 用户批准将轻量计划整理与并行输入校验分开、合并图片检查与编码、保留必要校验，并要求基于 xNav `glx` 创建独立分支供后续 AT 转换使用。

- 基线：`glx@a65de81a64af4129cfd404174a4c42be362adc46`。
- 分支：`perf/tracking-parallel-validation`。
- 修改范围：四视角入口的 `Stage4EpisodeSource`、`scan_stage4_inventory()`、`_load_stage4_episode()`、`_process_stage4_episode()`、`encode_video_from_paths()`、`convert_stage4_dataset()`，以及对应测试和说明。
- 数据合同：episode/task/frame 编号、完整帧数、四路视频、相机元数据、累计 nominal pose、等待帧和动作语义保持一致。训练端 H16 重采样不属于该转换器。
- 当前 DT 使用的主工作区未修改；合并及后续正式 AT 转换由用户安排。

验证结果：

- `test/test_tracking.py` 与 `test/test_tracking_stage4_parallel.py` 共 29 项通过。覆盖真实四视角视频转换、两个 worker 同时执行输入校验、每张输入图片只打开一次、相机路径按 episode 缓存、强制逆序完成时所有输出字节一致，以及 18 类损坏/不一致输入阻止发布。
- 异常覆盖缺帧、坏 JPEG、错误尺寸、帧号/时间/episode 身份错位、后续指令变化、非法动作、帧数不符、视角混淆、重复图片、相机引用冲突、绝对/越界路径和符号链接越界。额外确认失败时已有输出保持原样。
- 用相同的两条合成 episode 对照基线原版与本分支：2 episode、2 task、6 frame，16 个输出文件逐字节一致，包含 Parquet、8 个视频和全部元数据；其中零动作产生的重复位姿保留。
- 尚未测量真实全量 AT 的转换吞吐，不将合成测试作为生产加速倍数证明。

复现针对性测试（将 `TMPDIR` 指向任务独占目录）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q test/test_tracking.py test/test_tracking_stage4_parallel.py
```
