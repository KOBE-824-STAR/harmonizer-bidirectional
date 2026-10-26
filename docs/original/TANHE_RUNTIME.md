# tanhe-server 原生运行环境与真实模型验证

2026-10-21，目录 `/data/juziwei/hlk/double_DiT`。

单向读取、联合注意力、特征调制三种方案均已通过真实官方预训练模型的 **512×288、双视角、单卡 T1/T5 验证**。没有启动几何方案或正式长训练。完整证据见 `reports/tanhe_validation_summary.json`。

## 原方案保存情况

- 原模型 `double_dit/model.py`、原训练/启动脚本、原配置和文档均保持原字节；原始 17 个文件的 SHA256 已再次核验。
- 原双向方案的完整第 1000 步检查点仍为 `runs/train/latest.pt`，大小 8,057,053,230 字节，未改写。
- 另增原源码备份 `backups/baseline_before_tanhe_20261021/original_source.tar.gz`，同目录保存检查点位置记录。
- 旧检查点来自 1024×576 训练。本次三个验证均从官方预训练权重独立初始化，不将旧检查点改成新的结构。

## 本机环境

使用 `tanhe`（UID/GID 1000），按项目目录所有者核验运行身份，不使用 root 或 Docker，也不修改系统环境和既有 recondrive 环境。

- Python：`runtime/tanhe/venv/bin/python`，3.10.20。
- PyTorch：2.7.1+cu126；torchvision：0.22.1。
- Transformer Engine：2.1.0；扩展用本机 CUDA 12.8 工具链编译。
- 最终 `pip check` 通过（No broken requirements found）；完整版本锁定清单为 `runtime/tanhe/requirements_resolved.txt`，安装脚本为 `scripts/install_tanhe_dependencies.sh`。三种 GPU 验证后补齐了未启用的 Megatron 辅助依赖；已有依赖被约束为原验证版本，随后重新通过真实模型导入检查。
- Cosmos、imaginaire、multiview_harmonizer、VAE checkpoint 辅助代码来自原已验证环境，位于 `runtime/tanhe/vendor/`。
- Cosmos 正在使用真正的 0.6B Text2Image DiT、官方 tokenizer 与预训练权重，没有用微型替身执行本节验证。
- imaginaire 的两个未启用的 Qwen 文本编码器引用改为延迟导入，避免图像模式被额外文本模型的 FlashAttention 依赖阻断。模型数学运算未改动。补丁为 `scripts/lazy_text_encoder.patch`，原源码哈希记录为 `runtime/tanhe/vendor_origin_sha256.json`。

本机新增 `double_dit/model_tanhe.py`、`double_dit/model_injection_tanhe.py`、`scripts/train_tanhe.py` 和 `scripts/run_tanhe_validation.py`，保留历史入口。新验证入口只允许 `smoke/native`，不启动正式训练。

## 已修复的数据和权重路径

新数据清单：`data/tanhe_restored/sequences.jsonl`。

恢复了全部 448 条原序列的 2688 张 input/gt 图像，其中 input 对应模型输入，gt 对应原训练监督。依据 `selected_data/samples.jsonl` 的已登记 SHA256，从原机器可访问的上游图像恢复；源端与目标端逐文件核验一致。原 train/val 划分、相机内参、共同参考位姿、时间戳及序列结构保持不变，旧路径另记入 `source_original/target_original`。历史清单 `runs/data_v1/sequences.jsonl` 未修改。

新清单 SHA256：`7b3b5910ecf2b026350676ef68673be9a212d083f3be5f9df1180a19c59e559e`。

核心权重直接使用同服务器原有 harmonizer 目录，三者完整 SHA256 均通过原登记值校验：

- Harmonizer：`/data/juziwei/hlk/harmonizer/models/diffusion_harmonizer.pkl`。
- Cosmos DiT：`/data/juziwei/hlk/harmonizer/src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image/model.pt`。
- Tokenizer：上述 Cosmos 目录下的 `tokenizer/tokenizer.pth`。

原训练损失仍为 RGB MSE + 0.3×截断 VGG-LPIPS。公开 VGG 下载站在此服务器不可达，因此从原环境导出了实际参与损失的 26 个 VGG 特征参数及 5 个 LPIPS 线性参数，省去未使用的分类头。文件为 `runtime/tanhe/lpips_vgg_state.pt`（58,874,686 字节）。加载器 `double_dit/perceptual_tanhe.py` 核验导出文件哈希、原 VGG 哈希、LPIPS 原文件哈希，严格加载全部 VGG 特征参数并逐张量核验。构造时 `pnet_rand=True` 仅用于跳过网络下载，随后立即覆盖为已核验的预训练参数；计算损失时不使用随机 VGG。

## 三种真实模型验证结果

三种模式均 `PASS`，各执行 4 个诊断优化步。样本来自原 train 划分的同一真实序列 `0f1e7bcb4fadbea076a82cf6`，帧 69、72，各有两个相机。前两步走真实 T1→T5 生成历史；后两步按既有 smoke 协议复用单帧作梯度诊断，不伪造后续帧。

共同通过的检查：

- 官方 DiT 的 407 个参数张量、597,536,000 个参数元素覆盖率均为 100%，无核心缺失、形状不匹配或多余加载键。
- 当前输出有限，真实生成历史反馈到下一时刻；改变历史会改变当前输出。
- 主路首层与 VAE 参数实际更新，条件路获得非零梯度，梯度保持有限。
- 同步置乱相机输入、标定与身份后，输出置乱等价；最大差为 0。
- 开放诊断门后，改变另一相机的输入会影响目标相机输出。

分别结果：

- `one_way`：T1/T5 零门控相对官方单路最大差均为 0。测试峰值已分配显存约 12.68 GiB。
- `joint`：原生联合注意力前向/反向通过。初始 T1/T5 相对官方单路最大差为 0.0859375 / 0.07421875；该模式本就不要求零门控单路等价。测试峰值已分配显存约 12.43 GiB。
- `modulation`：T1/T5 零初始化相对官方单路最大差均为 0。测试峰值已分配显存约 12.66 GiB。

显存仅描述本次双视角诊断，不能外推为六相机训练显存或模型质量排名。新环境下另外复跑了 9 项 CPU 模块/接线测试，全部通过。

结果目录：

- `runs/tanhe_one_way/native/`
- `runs/tanhe_joint/native/`
- `runs/tanhe_modulation/native/`

每个目录包含 `RESULT.json`、`config.json`、`LAUNCH.json`、`gradient_audit.json`、`history_audit.json` 和 `rank0/official_load.json`。日志为 `logs/tanhe_<mode>_native.log`。

## 使用入口

在服务器项目目录执行：

```bash
runtime/tanhe/venv/bin/python scripts/check_tanhe_runtime.py
runtime/tanhe/venv/bin/python -m unittest discover -s tests -p test_injection.py -v
```

本次已执行的原生验证命令是：

```bash
python3 scripts/run_tanhe_validation.py --mode one_way --gpu 0 --phase native
python3 scripts/run_tanhe_validation.py --mode joint --gpu 1 --phase native
python3 scripts/run_tanhe_validation.py --mode modulation --gpu 2 --phase native
```

成功结果已经存在时，启动器会拒绝覆盖；请保留已验证记录，后续改版使用新的输出目录。启动器在运行前检查 GPU 占用和同模式运行锁，并在配置中记录实际用户、GPU、数据清单哈希和实现哈希。

本次尚未进行多卡 DDP 验证、正式训练、六相机验证或质量指标评测。几何方案仍等当前合成图像的深度与可见性数据补齐后再运行。旧 `run_experiment.py` / `run_injection.py` 的 Docker 路径属于历史环境，本机验证应使用 `run_tanhe_validation.py`。
