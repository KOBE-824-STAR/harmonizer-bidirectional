# 双路 DiT 实现接口与验收

## 实施边界

在本目录新建独立包，不修改原 `Scheme_A&Scheme_B`、`diffusionharmonizer_multiview`、官方权重、数据或历史实验。初版从官方单路 Harmonizer 构建，明确不加载 T4/A1/B1 已训练扩展；可以复用经核查的几何路由算法，但不能假设其 checkpoint 与新条件路兼容。

当前实现目录如下。`model.py` 合并了构建、交互、几何输入和 wrapper；数据准入和运行配置解析分别由两个脚本承担。运行范围与待完善的通用推理/评估接口见 [RUNNING.md](RUNNING.md)。

```text
double_dit/
  model.py             官方构建、双路交互、几何、VAE 及官方主损失
  data.py              图像与标定缩放、生成历史打包及量化
scripts/
  prepare_data.py      元数据、资产身份、标定与时间审计
  train.py             契约验证、生成历史训练和完整状态恢复
  run_experiment.py    GPU 与运行身份检查、容器、解析配置
  pipeline.py          四卡验证成功后启动正式训练
```

不直接继承当前 `TemporalMultiViewPix2PixTurbo.freeze_official_components()`：它只允许 `mv_adapters` 训练，会冻结主路和新分支。新 wrapper 从第一步起管理官方主路全量训练、VAE 的 `time_conv` 例外及新增模块，并在每次 `.train()` 后复核范围。参考配方和代码证据见 [TRAINING_REFERENCE.md](TRAINING_REFERENCE.md)。

训练前确认实际 tokenizer 的 encode/decode 允许反向传播。有些缓存 runtime 的推理接口带 `no_grad`，仅修改 `requires_grad` 不会恢复图；应对接经过输出及梯度核查的可训练接口，不得把可训练 VAE 悄悄冻结以绕过问题。共享 inner VAE 必须注册到 DDP；按参数对象去重，避免 pipeline 和单独 VAE 引用导致重复优化。冻结的损失网络不进入优化器，但不切断对预测图的梯度。

## 数据契约

每个序列的 manifest 必须包含 `split, scene_id, sequence_id, asset_set_id, render_variant_id`。每个相机观测包含 `camera_name, sample_data_token, timestamp_us, source_path, target_path, asset_mask_path, K, T_ego_from_cam, T_world_from_ego, image_hw`，以及资产身份和同步关系的来源证据。

目标路径和监督 mask 应由训练 dataset 单独返回为 `supervision`；推理 dataclass 不包含这些字段。GT 文件即使存在于同目录，也不能被推理 loader 隐式扫描。

按 scene 划分数据，禁止相同原始场景或同一生成样本的不同裁剪跨 train/val/test。相机 ID 先从来源的语义映射解析，再转换为统一 `camera_name`，不能直接复用不同项目的 `c0…c5` 数字顺序。本地分组报告的 `c1` 和模型 `CAMERA_ORDER` 的第二个位置含义不同，需显式映射。

需要重新生成准入报告，按 1–6 个有效视角分别统计样本/序列数量、资产一致性、可用几何边、时间间隔和 split。现有 `nusc_pair_grouped6/summary.json` 记录 58,151 组中仅 1 组完整六视角；`GROUPING_REPORT.md` 还说明源相机同步不保证同一生成资产集合。该审计是其指定数据快照的结果，不代表所有其他数据目录也只有一组。

同一个 asset_set_id 不意味着每个相机都必须看见全部资产，允许真实遮挡或不在视锥内；但必须证明来自同一世界状态和渲染版本。不能把缺测相机、另一资产版本、其他时刻 GT 当作参考补齐。2–5 视角子集需经过同等身份核查，数量不可从旧六视角报告直接推断。

时间同步容差、最大历史间隔和有效序列长度必须从数据时间戳统计后填写。配置保留这些值为 `null`，运行校验器应拒绝未填写的数据协议，不能默认接受任意时间差。

## 模型构建顺序

1. 创建原 0.6B Text2Image DiT 和官方 tokenizer，严格加载 Harmonizer，并记录实际权重 hash、张量名/形状和覆盖率。方案配置中的预期 hash 来自本地注册表，正式运行需重新计算。
2. 断言 `20/1280/20/16/patch2/temporal_patch1` 契约；拒绝替换成 2B、其他多视角模型或未匹配的 checkpoint。
3. 保留主路全部原始结构；独立克隆 x_embedder、20 个 block、噪声 embedding、必要 norm/condition projection 等供条件路使用。条件路不持有未使用的 final_layer。
4. 比较克隆后的对应初值相等，并检查参数存储互不别名。仅 tokenizer 权重共享；不能用同一个 block 对象重复挂载为两路。
5. 条件路对当前图像 T1 latent 使用与主路一致的输入缩放。先原 patch embedding，再空间池化及相应位置编码，添加条件路的相机/几何特征。
6. 创建 5 组条件路跨视角模块、5 组 M→S 和 4 组 S→M 模块；归一化和投影参数各自独立。对原主路不注入无门控的相机 embedding，以保留关闭新模块时的基线等价性。
7. 原官方 condition、噪声 embedding 的数值语义及输出参数化保持一致，decoder 按官方训练范围更新；保存架构签名 `double_dit_v1_full`、训练配方签名 `harmonizer_official_joint_v2`、配置 schema 2、模块版本及主/条件路/VAE 参数清单。

主路从普通 Harmonizer 开始。初始化时关闭交换模块，应与声明的官方主路及 VAE 在同一输入上等价；从第一次更新起主路和 VAE 就会改变，关闭分支不再等于初始官方权重。旧“冻结预热加末五层微调”配方不能以同架构名静默恢复到本配方，必须核对训练签名、参数组及 optimizer/scheduler 状态。

保存主路、条件路、交互、共享 VAE、优化器、scheduler、step、随机数状态及数据/历史协议，不能只保存新增分支。以官方 checkpoint 初始化的新训练创建新的优化器状态；完整恢复则校验并恢复全部训练状态。VAE 已可训练，不得跨更新复用旧 latent 缓存；历史保存 RGB，当前步重新编码。

## 交互调度伪代码

以下是接口说明，不是可执行模型实现。

```python
S = main_embed(precondition(main_temporal_latents))
M = condition_embed_and_pool(precondition(current_T1_latents))
M = add_camera_and_geometry(M, calibration)

for i in range(20):
    S = main_blocks[i](S, official_noise_condition, official_context)
    M = condition_blocks[i](M, official_noise_condition, official_context)
    if i in [3, 7, 11, 15, 19]:
        M = routed_view_exchange[i](M, calibration, valid_views)
        S_current_before = S[:, :, current_latent_slot]
        M_before = M
        delta_S = read_M[i](S_current_before, M_before)
        if i != 19:
            delta_M = read_S[i](M_before, pool(S_current_before))
            M = M_before + tanh(gate_MS[i]) * delta_M
        S = functional_update_current_slot(
            S, S_current_before + tanh(gate_SM[i]) * delta_S)

F = original_head_and_unpatchify(S)
z_hat = original_c_skip * original_input_latents + original_c_out * F
return decode_current_only(z_hat)
```

这里的视角维度和 temporal 维度是显式布局；调用原 block 前根据其接口重排，但不得把其他相机变成同一个时间序列。不要对有梯度的 view 做危险的原位赋值。条件路与主路在每个交换点使用各自实际 token 网格，padding 和池化边界必须有一致的位置与有效性掩码。

训练不另设条件路输出头。对每个有效相机的当前输出及 GT 在 `[-1,1]` 范围计算官方 MSE 和截断 VGG-LPIPS，反向到主路、条件路、交互模块及共享 VAE。只在训练优化路径使用 LPIPS；原训练脚本附带的验证 LPIPS 不能直接复用。

训练图像先计算监督，再将独立的 detach 副本量化并写入历史。多视角有效数量不同时，按所有参与更新的有效当前图像数归一化；DDP 下需要考虑各 rank 样本数差异与梯度平均，不能简单平均每张卡的局部均值。官方 `LOSS_INPUT_ONLY` 应沿时间轴选当前槽，再处理全部有效相机。

## 自回归驱动伪代码

```python
for sequence in dataset.sequences_sorted_and_isolated():
    history = HistoryBank(sequence.identity, max_length=4)
    for timestamp, observations in sequence.in_time_order():
        # 只读取当前输入；检查同一组内曝光时间与资产身份。
        batch = assemble_current_valid_views(observations)
        packed = []
        for camera in batch.camera_names:
            previous = history.get_before(camera, timestamp)
            if not previous:
                packed.append(T1(batch.source[camera]))
            else:
                refs = previous[-4:][::-1]  # 最近在前
                while len(refs) < 4:
                    refs.append(refs[-1])  # 重复最早的已有输出
                packed.append(T5(batch.source[camera], refs))
        # 对 T1/T5 主路分桶执行，当前槽重新汇合供跨视角交互。
        # 不丢弃缺历史相机的当前观测，不向 temporal latent 填假相机。
        predictions = model(packed, batch.current_sources, batch.calibration)
        committed = save_pngs_and_journal_atomically(predictions, timestamp)
        history.append(reload_saved_rgb(committed))
```

混合 T1/T5 时，原主路 block 按 temporal 长度分桶，当前 token 在交换点汇合，然后返回各桶继续执行。条件路当前 T1 始终在同一个有效视角集合中。初版可先实现各相机历史齐备的组，但必须明确筛选规则和覆盖率，不能悄悄丢弃难样本。

日志记录 sequence/sample/camera、真实时间戳、每个历史输出路径与 SHA256、重复标记、输入 hash、模型 hash、配置 hash。恢复时逐项核验；输入、模型或历史链改变后，依赖它的后续输出必须失效重算。不得只凭 PNG 文件存在就跳过。

## 实施顺序与验收

| 顺序 | 交付 | 必须通过的检查 |
| --- | --- | --- |
| 1 | 官方训练配方及数据准入 | 命令和代码 hash；配对目标语义、来源、视角身份、几何和时间完整；运行配置不含未解析字段 |
| 2 | 双路构建和交换模块 | 权重独立；门控关闭等价；正确 block 数量；无未消费的最后反向模块 |
| 3 | 单时刻多视角前向 | 1/2/6 视角、缺测、无重叠、标定缺失、池化边界均按契约处理 |
| 4 | 自回归驱动 | 第二个时刻开始读取首个生成结果；无未来/GT；混合 T1/T5 与恢复链正确 |
| 5 | 官方主损失与联合优化 | MSE/截断 LPIPS 数值和图像范围一致；从首步主路全量及 VAE 官方范围参与优化；共享参数去重 |
| 6 | 短训练和同协议评估 | 全部九项状态明确；消融预算与输入序列一致；再决定是否扩大训练 |

关键测试应验证行为而非重复实现：

- 更换未来帧和 GT 文件，固定当前及过去输入，当前输出不变。
- 给第二帧提供不同的合法首帧生成历史，读取日志和特征路径必须反映变化；在有训练权重后检查输出对历史具有实际依赖。
- 置乱相机 tensor 顺序并同步置乱名字/标定，输出应仅相应置乱，而不是改变相机语义。
- 有效的无邻居视角不产生 NaN，非法标定明确失败；被 mask 的源视角扰动不影响输出。
- 初始化时关闭新模块，与同一官方主路和 VAE 在相同 T1/T5 输入上的输出一致；使用同一目标时，主损失及主路/VAE 梯度也应对应一致。FP32 用严格数值容差，BF16 根据算子路径给出实测容差。
- 从第一个训练步检查主路早层、中层、末层、embedding 和输出头的参数均在优化器且可训练；用实际梯度及更新证据确认不是仅末五层在学习。VAE encoder/decoder 的可训练参数得到梯度，`time_conv` 和感知网络保持冻结；个别未激活参数不强求单步变化。
- 沿外层门控、条件路、反向交换、跨视角路径依次检查新增梯度。串联零门控可延后深层梯度；在多个有效更新后仍断路则判为失败，不能把长期不学习归因于正常预热。
- 用相同 pred/GT 验证官方 MSE 与 LPIPS 的值域、逐样本 clamp、权重和 reduction；超过 clamp 上限的样本应具有与参考一致的梯度。该项是训练损失契约检查，不输出质量指标分数。
- 检查训练与恢复配置没有冻结阶段，附加 L1 权重为零，学习率为选定配方值，梯度裁剪覆盖所有可训练参数，checkpoint 包含更新后的主路和共享 VAE。
- 若使用 activation checkpointing，backward 重算时保留同一几何路由、条件和 token 元数据，禁止读取已退出的临时全局状态。
- 启用条件路时必须有跨视角信息实际影响主路的证据；零门控、未训练模型不能作为双路效果结论。

所有模型实验、数据生成、训练、推理、评估均以 `hlk` 执行。GPU 启动前检查指定卡的可用性和重复进程，输出、日志、checkpoint 和 tmux 归 `hlk:hlk`。正式启动后报告 trainer/worker 所属用户、GPU 使用、tmux session、log/output 路径及当前 step。方案编写阶段不需要启动实验。

## 本地证据

以下是本方案核查过的本地实现与报告，适用范围分别列出；旧报告中的实验表现不作为新双路结果。

| 路径 | 支持的事实 |
| --- | --- |
| `/mnt/hlk/workspace/harmonizer_official_recipe_20261012/upstream_reference/README.md` | 所选官方推荐训练命令：全量主路、250、2e-5、MSE 加感知项 |
| `/mnt/hlk/workspace/harmonizer_official_recipe_20261012/upstream_reference/src/train_pix2pix_turbo_harmonizer.py` | 优化器参数范围、训练前向、当前槽监督和损失计算 |
| `/mnt/hlk/workspace/harmonizer_official_recipe_20261012/upstream_reference/src/pix2pix_turbo_harmonizer.py` | `set_train`、VAE 的 time_conv 冻结规则、训练共享 forward |
| `/mnt/hlk/workspace/harmonizer_official_recipe_20261012/upstream_reference/src/utils/training_utils.py` | 命令未覆盖的默认参数、图像值域与目标字段 |
| `/mnt/hlk/workspace/Scheme_A&Scheme_B/multiview_harmonizer/runtime_contract.py` | 原 0.6B DiT 的层数、宽度、头数、通道和 patch 契约 |
| `/mnt/hlk/workspace/Scheme_A&Scheme_B/upstream/harmonizer/src/pix2pix_turbo_harmonizer.py` | 一次 `denoise(...).x0`、原编码/解码缩放与可选 skip 缓存 |
| `/mnt/hlk/workspace/Scheme_A&Scheme_B/multiview_harmonizer/wrapper_temporal.py` | `c_in/c_noise/c_skip/c_out`，主路包装与冻结范围 |
| `/mnt/hlk/workspace/Scheme_A&Scheme_B/multiview_harmonizer/layout_temporal.py` | 时间/相机布局、当前槽 index 0、当前优先打包约定 |
| `/mnt/hlk/workspace/Scheme_A&Scheme_B/multiview_harmonizer/factory.py` | 先严格加载官方权重再附加模块的构建顺序 |
| `/mnt/hlk/workspace/Scheme_A&Scheme_B/multiview_harmonizer/constants.py` | 官方路径、预期 hash、模型相机顺序 |
| `/mnt/hlk/workspace/Scheme_A&Scheme_B/configs/scheme_B_TMV_pose_routing.json` | 本地几何路由参数、RGB/latent 长度关系 |
| `/mnt/hlk/workspace/diffusionharmonizer_multiview/reports/load_audit_A.json` | 本地旧加载审计中的 597,536,000 参数记录 |
| `/mnt/hlk/workspace/harmonizer_autoregressive_20261015/infer.py` | 使用已保存预测历史的旧驱动；前四次 T1，与本方案新启动协议不同 |
| `/mnt/hlk/workspace/nusc_pair_grouped6/GROUPING_REPORT.md` | 指定数据快照的同步、资产身份与六视角覆盖限制 |
| `/mnt/hlk/workspace/nusc_pair_grouped6/summary.json` | 该快照 58,151 组、1 组完整六视角 |
| `/mnt/hlk/AGENTS.md` | 运行用户、自回归和固定九项指标要求 |

官方训练机制的证据以本目录 TRAINING_REFERENCE 为准；双路交互层位、新模块学习率继承、跨视角批组织和生成历史是扩展选择，不是官方数据流程或已验证的双路效果。
