# 双路 DiT 实验运行说明

## 训练尺寸更新（2026-10-21）

后续训练固定为 **512×288（宽×高）**，即 `configs/double_dit_v1_full.json` 的 `data.image_hw=[288,512]`；`data.evaluation_hw` 同为 `[288,512]`。数据加载器按该尺寸缩放输入与监督，并同步缩放相机内参。启动器将 `autoregressive.history_image_hw` 设为训练尺寸，历史继续来自此前生成结果，经相同尺寸 uint8 往返后反馈，首帧 T1、后续 T5。

以下已完成实验及 demo 是原 1024×576（宽×高）的历史记录。`runs/train/latest.pt` 仍为该尺寸训练至第 1000 步的完整 checkpoint，原运行配置与记录不改写。本次只更新后续配置，没有启动 512×288 训练或验证。启动器会拒绝复用不同尺寸的 native/DDP 验证结果，也会拒绝将已有训练目录直接按另一尺寸续跑；新尺寸实验须独立存放原运行记录，并重新完成对应尺寸验证。

实验使用 GPU 0–3，运行用户为 `hlk`。主路 20 层、独立条件路 20 层及共享 VAE 按方案联合训练，VAE 的 `time_conv` 保持冻结；官方 MSE 加截断感知损失，新增门控为零初始化，正式训练重新从官方权重开始，不复用 smoke 更新后的参数。

## 实现与验证

- `double_dit/model.py`：严格加载、双路交互、几何路由、可训练 VAE 和官方损失。
- `double_dit/data.py`：图像读取、相机缩放、生成历史打包和量化。
- `scripts/prepare_data.py`：来源、资产身份、标定、曝光时间及图像有效性审计。
- `scripts/train.py`：单卡契约检查、四卡梯度同步、训练状态、完整 checkpoint 及恢复接口。
- `scripts/run_experiment.py`：GPU 与重复容器检查、运行配置、hlk 容器及日志。
- `scripts/pipeline.py`：四卡短验证通过后才进入正式训练。

小分辨率验证结果在 `runs/smoke/RESULT.json`。576×1024 原生分辨率检查在 `runs/native/RESULT.json`：T1/T5 零门控最大差均为 0，视角置乱等价检查最大差为 0，改变生成历史会改变当前输出。主路首层和 VAE 参数实际更新，新增条件路和两方向注意力均已观察到非零梯度。跨视角作用检查使用人为开放的 0.2 诊断门控，仅用于证明代码路径，不代表未训练零门控模型已获得多视角质量提升。

四卡短验证和正式训练状态以对应目录的 `STATUS.json` 与 `RESULT.json` 为准；有 `FAILED*.json` 时应结合时间和日志区分历史失败尝试和后续成功结果。

## 数据范围

通过审计的候选是双视角，正式训练包含 388 条序列、588 个时刻组、1176 张相机图像；验证有 60 条序列、84 个时刻组、168 张相机图像。训练和验证分别来自 229 和 36 个互不重叠场景。训练有 117 条多时刻序列，验证有 19 条，其余为单时刻序列。该范围不能称为完整六视角训练。

资产身份及渲染契约通过来源 metadata 核对；相机内参来自 nuScenes，相机曝光的 ego pose 转到共同参考坐标。初始候选 677 组中，4 组曝光差超过 50ms、1 组缺少可核查的 exact-asset metadata，均被排除。历史间隔上限为 1 秒，轨迹还按资产、渲染版本和相机集合隔离；同一 source 轨迹不跨 train/val 场景拆分。

清单：`runs/data_v1/sequences.jsonl`。准入报告：`reports/data_audit.json`。时间是稀疏观测，不能称为连续原视频帧训练。

## 训练与历史约定

用户已将本轮正式预算调整为总计 1,000 optimizer steps，启动器默认使用 `--steps 1000`，到达后保存完整检查点并正常退出。原训练进程不支持热修改预算，因此从已核查的第 5 步完整检查点恢复，恢复权重、优化器、调度器及各 rank 随机状态；未保存部分重跑。原始训练配方中的 10,000 保留用于检查点兼容性，实际停止上限以命令行、`run_control.effective_total_steps` 和 `STATUS.json.total_steps` 为准。

每卡每次累积 2 个 rollout，四卡合计 8 个 rollout。按全局有效当前图像数归一化损失；轨迹长度不等，实际图像数会随更新变化。主路、VAE、条件路和交互模块学习率均为 2e-5，AdamW，constant scheduler，BF16。当前数据轨迹最长 4 帧，配置中的 5 帧 rollout 是上限，不补造未来帧。

实际执行的历史协议记为 `ar_generated_repeat_oldest_native_rgb_v1`：历史是当前模型此前输出的 detach 副本，在模型原生分辨率做 uint8 往返，像素等价于同分辨率 PNG 无损保存后读取；它在训练中存于内存以减少磁盘写入。首帧 T1，随后用已生成历史补足 T5，从不喂入 GT 或未来图像。历史记录中保存对应预测 hash 和时间身份。

这一原生分辨率历史缓存不同于“先缩到评估发布尺寸再读回”的方案；未来验证和对照必须沿用运行配置中的实际协议，不能沿用另一种历史分辨率的分数。当前监督在未量化输出上计算，历史副本才量化。训练 rollout 内先逐帧累积梯度，完成该次更新所需 rollout 后再更新参数。单卡 smoke 的逐步优化仅用于梯度诊断，不作为正式训练协议或质量评估。

同拓扑下，当前实现按轨迹固定的有效相机集合分组；所有相机在轨迹内有相同历史长度。尚未提供一个 batch 内任意混合 T1/T5 相机的通用推理入口，也未实现完整九项评估脚本，不得将训练验证结果表述为正式质量结果。

## 会话与输出

tmux socket 为 `double_dit`，session 为 `train_0_3`。以 hlk 身份查看：

```bash
tmux -L double_dit attach -t train_0_3
```

管线日志：`logs/pipeline.log`；四卡短验证：`logs/ddp_smoke.log`；正式训练：`logs/train.log`。训练输出：`runs/train/`，逐 rank 历史：`history_rank*.jsonl`。第 5 步保存首个完整恢复点，此后每 500 步及结束时原子更新 `latest.pt`；临时文件只用于完整写入后替换，不保存数百份完整模型。

恢复命令在主机上以 hlk 执行，需有可访问 Docker 的进程权限：

```bash
python3 scripts/run_experiment.py train --steps 1000 --resume /mnt/hlk/workspace/double_DiT/runs/train/latest.pt
```

恢复核对训练配方、实现 hash、数据 hash 和全部参数状态；历史在 optimizer-step 边界重置，旧日志保留。初始加载使用 `harmonizer_official_recipe_20261012/weights/` 下与注册哈希一致的原官方权重，不修改该目录。启动器只向本实验目录写入。

## 固定 demo 场景生成（2026-10-21）

新增 `scripts/prepare_comparison.py`、`scripts/infer_comparison.py` 与 `scripts/launch_comparison.py`。使用正式 `runs/train/latest.pt` 的第 1000 步全部模型参数（严格加载，包括已训练 VAE），不改训练代码。选取既有 `2026-10-19` 对比中的五组场景帧及真实历史前缀，共 12 个时刻、72 张六相机原生输出，其中 30 张为指定对比输出。

原六相机数据目录已缺失；优先使用 SHA256 一致的已保存输入。历史输入可从旧 PNG 全相机比较图精确裁剪恢复，必须重新编码后仍匹配原始 PNG 的完整 SHA256 才能准入。相机曝光姿态从 nuScenes 原始标定与 ego pose 构建，不使用训练缓存中的原图历史。

输出：`/mnt/hlk/demos/harmonizer_comparison/2026-10-21/double_dit_step1000/`。入口为 `overview_full_images.png`、`overview_detail_crops.png`；每组另有包含旧九模型的 `comparison_all_models.png`。日志为该目录下 `inference.log`，状态为 `STATUS.json`，逐时刻反馈证据为 `history_audit.jsonl`。tmux socket：`double_dit_comparison`；session：`comparison_20261021`。启动器仅选择没有已有计算进程的 GPU，宿主启动器和容器用户均为 `hlk:hlk`。

严格采用训练的 `ar_generated_repeat_oldest_native_rgb_v1` 协议：576×1024 生成，首帧 T1，此后使用本模型已生成历史填充 T5；原生 uint8 往返作反馈，288×512 展示图不参与反馈。旧对比模型的启动和历史缩放协议不同，因此本次仅为同输入定性比较。模型训练使用双视角，本次六视角推理属于视角数扩展；未计算质量指标。
