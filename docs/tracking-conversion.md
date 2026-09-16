# Tracking 实测位姿转换

`tracking.py` 将 EVT-Collect tracking 数据转为 LeRobot v2.1。每一行仍对应动作前的 RGB 观测；`observation.state` 和 `action` 都存储**这一帧的实测位姿**，不再通过控制命令推算位置。

## 位姿和控制命令

- **STT**：位置来自 `teacher.robot_position`；朝向由同帧 `target_position` 和 `yaw_error_rad` 恢复。采集端的误差在 Habitat XZ 平面计算，因此输出朝向为 `atan2(-target_delta_z, target_delta_x) + yaw_error_rad`。
- **DT/AT**：直接读取 `teacher.dt_scene.robot_pose.position_m/yaw_rad`。不使用预测端点、上一转移或终端位姿冒充当前帧位姿。
- 所有位姿转换到第一帧机体坐标：X 向前、Y 向左、Z 向上；旋转存为 `qx,qy,qz,qw`，第一帧为单位位姿。真实高度变化保留。
- 原始命令保存在独立的 `tracking.command_normalized` 列（forward/left/yaw），不覆盖同名 `action` 的位姿合同。
- 缺实测字段、非有限位姿或 STT 目标与机器人重合而无法恢复朝向时，报告 episode/行号并终止转换；没有名义积分回退。

受阻时，非零前进命令可以对应零位移；滑动时，位姿保留实际侧向位移。等待帧和最后一个已有观测都保留，不额外生成没有 RGB 的终端帧。H16 空间重采样由 Enactive 训练端完成，本转换器不做重采样、减速或末尾样本筛选；VLN 转换不变。

## 输入与输出

优先直接读取 raw，避免 MP4 → JPEG → MP4 的中间过程：

```bash
python tracking.py \
  --raw-root /path/to/tracking/raw/stt \
  --output-dir /path/to/new/stt \
  --work-dir /path/on/same/filesystem/work \
  --workers 32
```

raw 目录需为 `<seed>/<scene>/<episode>/{status.json,camera.json,steps.jsonl,videos/}`。沿用 EVT-Collect 导出条件，只转换 `state=complete` 且 `success=true` 的 episode，日志报告跳过数量。四路视频直接复制，解码一次计算像素统计并核对帧数、时间戳、分辨率及编码合同，不重新编码。

已有 Stage-4 导出仍可通过 `--processed-root` 输入。其 manifest 的 `input_root` 必须指向对应原始数据；worker 按 episode 读取 raw steps，核对每行索引、时间和命令，再将实测位姿与 JPEG 序列配对。Stage-4 的图片校验和编码沿用原流程。旧 tar 输入只有在行内也具有实测 teacher 字段时才能转换，不再支持仅靠命令生成位姿。

单个训练数据根直接位于 `--output-dir`，**不再增加 `train/`**：

```text
stt/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/video.front/episode_000000.mp4
  videos/chunk-000/video.left/...
  videos/chunk-000/video.right/...
  videos/chunk-000/video.rear/...
  meta/info.json
  meta/episodes*.jsonl
```

原始 `back` 映射到 `video.rear`。相机元数据继续写入 extras；位姿来源标记为 `measured_first_frame_local_body_pose_v1`、`pose_is_executed=true`，同时记录具体来源字段。位姿、命令及图像统计均按本次实际输出重新生成；Enactive 的 H16 变换后统计仍需在新实验中单独计算。

## 并行与发布

`--workers` 控制 episode 并发；raw 每个 worker 同时解码一路视频，解码器单线程，四个视角依次处理。Stage-4 编码器同样单线程。编号与元数据顺序不依赖 worker 完成顺序。后续根据 Slurm 分配的 CPU、内存和共享存储吞吐确定 worker 数，不预设增加 worker 一定线性提速。

转换先写入 work 下的 staging，完成必要校验后重命名为正式输出，因此 work 与 output 必须在同一文件系统。错误时保留 staging，正式数据不发布；默认拒绝覆盖已有目录。新版本应写入新目录，保留原数据。

进度中的 `Converted N/M episodes` 包含已完成的位姿转换、视频处理与检查；`Validating final dataset metadata` 后成功退出并发布目录才代表完成。

## 针对性验证

```bash
TMPDIR=/path/to/task/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q test/test_tracking.py \
    test/test_tracking_stage4_parallel.py test/test_tracking_measured.py
```

覆盖 STT/DT/AT 坐标方向、跨 ±180°、受阻/滑动、缺位姿失败、命令留存、四视角 raw 和 Stage-4 转换、直接目录布局、并发确定性和坏输入阻止发布。真实数据只做小样本试转；不以这些检查证明训练收益或全量转换吞吐。
