# OVON 转换方案独立复审

复审对象：`ovon_conversion_plan.md`（对照上一轮 F1--F7）
复审结论：**CHANGES_REQUIRED，不允许开始代码实现**。

## 已解决的上一轮问题

- **STOP 合同**：已明确最后一条 JSONL 是最后视频帧对应的 `STOP=0`，不追加帧、不覆盖，并要求全量核对动作集合。
- **JSONL/NPZ 双源合同**：已明确 JSONL 为逐帧权威、NPZ 必须存在并逐元素比较；冲突阻断 episode。
- **视频 writer 契约**：已明确 `LeRobotCreator` 接收逐帧图像，必须解码 raw MP4 后提交，不能把 MP4 路径写入 parquet；`back` 到 `video.rear`。
- **路径安全**：已规定相对路径、拒绝绝对路径和 `..` 逃逸。
- **报告闭合**：已给出 source errors 与成功清单闭合公式，并禁止发布残缺 staging。
- **fixture**：已要求两个 shard、重复数字 episode id、真实 JSONL/NPZ、STOP=0 和四路短视频，以及输出集合/帧数/feature 校验。

## 仍需修订的阻断问题

### F8：稳定 episode index 与并行 writer 语义矛盾

方案同时要求 `--num-workers 8`、按 `source_episode_key` 字典序分配稳定 `episode_index`，但仓库 `LeRobotCreator` 的 worker 在 `WorkerEpisodeBuilder` 初始化时通过 metadata service 动态 `allocate_episode_index()`；提交顺序/worker 调度会改变分配顺序。因此仅“按字典序提交”不能保证稳定索引，且 plan 未规定如何把预分配 index 传给 writer。必须在方案中二选一并写清实现合同：

1. 首版禁用并行 episode writer（固定单 worker），按排序顺序提交；或
2. 修改/扩展 writer，使 episode index 显式由主进程预分配并传入，且恢复时验证映射。

### F9：task index 的“字典序分配”未落到 LeRobotCreator API

`add_task()` 是按调用顺序注册 task，plan 只写“按文本字典序去重分配”，却未规定转换前收集全部 task、排序后预注册，或改 writer API。需要明确预扫描并调用 `add_task` 的顺序，且并发时禁止 worker 隐式新增 task；否则 task index 不稳定。

### F10：fixture/静态验证尚未可执行地落地

方案要求 fixture 含短 MP4、NPZ 等，但未说明 fixture 位置、生成方式、是否纳入仓库、测试如何在无真实数据/无 GPU 下运行。需指定最小固定 fixture（例如 `test/fixtures/ovon_minimal/...`），明确用标准库/已有 OpenCV 或 ffmpeg 生成，测试命令不访问 `/data/glx/...`，并将 `LeRobotCreator` 的进程资源可靠关闭。否则“静态验证”无法复现。

### F11：报告字段与 skipped 语义仍有歧义

公式允许 `skipped_episodes`，但又写“任何 conversion error 都阻断发布”，未定义是否允许可恢复 skip。应明确首版 `skipped_episodes` 恒为 0；任何源校验/转换错误均 `conversion_errors` 并阻断，只有显式审计模式可输出报告而不发布。还需规定 `converted_episodes` 仅在 episode flush 成功后计数。

## 通过条件

补齐 F8--F11 后重新复审。当前 verdict 为 **CHANGES_REQUIRED**，不得开始实现。

## 第二轮独立复审（F8--F11）

复审对象：修订后的 `ovon_conversion_plan.md`。

### F8：episode index 稳定性

已解决。方案将首版明确限制为 `--num-workers=1`，并规定按 `source_episode_key` 字典序串行提交；同时保留输入、配置和 schema fingerprint 作为 resume context。该约束消除了 worker 调度导致的动态 index 非确定性，且没有引入未实现的 writer API。

### F9：task index 预注册

已解决。方案规定转换前预扫描全部可转换 source episode，按字典序唯一化 task 文本，并由主进程按该顺序调用 `creator.add_task(task)`；worker 只能使用已注册整数，禁止隐式新增。该流程与现有 `add_task()` 调用顺序语义一致。

### F10：fixture 与静态验证可执行性

已解决。方案固定 fixture 路径为 `test/fixtures/ovon_minimal/`，测试 helper 在临时目录生成两个 shard、2×2 两帧 H.264 MP4、JSONL 和 NPZ；测试不访问真实 `/data/glx/...` 数据且不依赖 GPU。方案还要求 finally 中执行 `wait()`/进程清理，并将视频编码不可用视为明确失败而非 skip，因此验证可复现且不会掩盖环境缺陷。

### F11：skipped 与错误闭合

已解决。方案明确首版 `skipped_episodes` 恒为 0；任一源校验或转换错误均计入 `conversion_errors` 并阻断发布，仅 `--audit-only` 可输出不可发布报告。`converted_episodes` 仅在 episode flush、metadata 和视频编码全部成功后计数，闭合公式和 staging 发布条件一致。

### 第二轮结论

F8--F11 均已得到可执行的实现合同，未发现新的阻断问题。**verdict: APPROVED**。方案审核通过，可以开始代码实现；实现完成后仍需进行独立代码审核及静态验证，且不得在本任务中发布数据或启动训练。
