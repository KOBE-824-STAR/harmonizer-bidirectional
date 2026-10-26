# 官方训练参考与双路扩展边界

本方案采用本地保存的 DiffusionHarmonizer 官方推荐训练配方，主路从发布的 Harmonizer 权重初始化，按官方范围继续训练。条件路与其联合优化，不设主路冻结预热。证据日期为 2026 年 10 月 20 日；这里只核查源码和配置，没有执行模型或验证训练结果。

## 参考版本与证据

参考根目录为 `/mnt/hlk/workspace/harmonizer_official_recipe_20261012/upstream_reference`。相邻 `SOURCE_PROVENANCE.json` 将其登记为 upstream commit `32f3eca8c0a144d9067430f9be07de7723ef172b`；下列文件本次计算的 SHA256 均与该登记一致。这是本地版本证据，不代表已联网核查最新上游。

| 文件 | SHA256 |
| --- | --- |
| README.md | `3da75cc48df41a7f32db9172cf9b183ea81045852d4754751c3911d227af664a` |
| src/train_pix2pix_turbo_harmonizer.py | `cf22133e223d87061b48457f5508a4aa58371e67a5bdd4e753f06afd88f03b5a` |
| src/pix2pix_turbo_harmonizer.py | `14e23e3f8ec06d0241ce60a40ad953bc4bf495486d0972a9c6df87849efbb6cf` |
| src/utils/training_utils.py | `e35f825f29e3896784aff882346436a463999b7e0ab82f700af76e14ddac9db9` |

| 项目 | 核查结果与位置 |
| --- | --- |
| 推荐命令 | [README](../harmonizer_official_recipe_20261012/upstream_reference/README.md) 的 Multiple GPU Training Command：`train_full_unet`、`timestep=250`、LR `2e-5`、10,000 steps、BF16、`576×1024` |
| 主路范围 | [模型代码](../harmonizer_official_recipe_20261012/upstream_reference/src/pix2pix_turbo_harmonizer.py) 的 `set_train()`：主路全部 `requires_grad=True`，该版本未实现 partial UNet 训练 |
| VAE 范围 | 同一函数默认启用 VAE 参数训练，仅对名称含 `time_conv` 的参数冻结；推荐命令没有 `freeze_vae` 或 `freeze_vae_encoder` |
| 优化器范围 | [训练器](../harmonizer_official_recipe_20261012/upstream_reference/src/train_pix2pix_turbo_harmonizer.py) 的 `layers_to_opt` 收集主路及未冻结 VAE，AdamW 使用 CLI 配置 |
| 训练前向 | 同训练器调用 `net_pix2pix(x_src)`，模型 forward 编码输入、取固定 `self.timesteps/1000`、一次 `denoise(...).x0`、解码；没有随机噪声注入或 timestep 抽样 |
| 目标和范围 | [数据与参数代码](../harmonizer_official_recipe_20261012/upstream_reference/src/utils/training_utils.py) 将 source/target 归一化到 `[-1,1]`，目标来自 `output_pixel_values` |
| 损失定义 | 训练器 `LOSS_INPUT_ONLY=True` 后使用当前槽 RGB MSE，以及 VGG-LPIPS 输出先 clamp 到 `[0,0.3]` 再均值 |
| 损失权重 | README 命令 `lambda_l2=1.0`、`lambda_lpips=0.3`，GAN/Gram/CLIP similarity 为 0；optical-flow 由 parser 默认 0 |
| 优化器默认 | parser：betas `(0.9,0.999)`、eps `1e-8`、weight decay `0.01`、clip `1.0`、scheduler `constant` |

推荐命令优先于未覆盖的 parser 默认值：例如 parser 的 `timestep=999`、`lambda_lpips=5`、LR `5e-6` 均被命令覆盖。`use_sched` 是模型标志，不等同于 optimizer 的 `lr_scheduler`；该版本 forward 没有依据它执行多步扩散。constant 学习率调度也不能因为存在 `lr_warmup_steps=500` 参数就描述为有效预热。

源代码中存在 GAN、Gram、CLIP、光流和验证 LPIPS 路径，不意味着推荐配方启用了这些项。新实现不照搬未使用的辅助网络加载和验证指标代码。运行时模型迁移仍需严格权重审计；不能复制旧 `strict=False` 调用后忽略实际缺失的 live 权重。

## 保留的官方机制

主路全部参数及 VAE 的官方可训练子集从第一步参与优化；保留 latent 缩放、preconditioning、固定训练 timestep 的已核查语义、官方输出头、当前图像预测目标、MSE 和截断感知损失、原 optimizer 超参数与 scheduler。新增分支没有独立预测任务，其参数通过同一个主损失学习。

LPIPS 只按本次用户要求保留为官方训练损失，不用于验证/测试评分；其训练 loss 日志和九项质量评估表分开。可训练 VAE 需要真实的 encode/decode 梯度、DDP 参数注册及共享参数去重，不能仅凭 `requires_grad` 声称已实现官方更新范围。

## 明确属于扩展的部分

- 两条 DiT、池化条件 token、几何路由、五个层级交互和零门控初始化。
- 可变相机集合及外层多视角损失平均；不把官方“当前槽”解释成“第一个相机”。
- 生成历史的自回归 rollout、历史不足时补齐、量化和 detach；这改变了官方原始数据/参考构造方式。
- 新增参数沿用 `2e-5` 的学习率，以及多视角 rollout 的 batch 组织；它们没有官方双路实验结论支持。
- 数据准入审计、严格权重加载和完整状态恢复；任何适配必须保持已选训练目标与参数范围。

因此名称使用“基于官方训练机制的双路扩展”。训练配方签名为 `harmonizer_official_joint_v2`，架构名仍为 `double_dit_v1_full`，因为交互拓扑未改变。新增资产 L1、latent L1 等只允许作为有明确标识的后续损失消融，默认权重为零。
