# 六相机双向交流 Harmonizer

此私有仓库保存 **2026-10-24 完成的 bidirectional 第 1000 步实验**的代码、配置、训练记录和完整恢复权重。它是基于官方 DiffusionHarmonizer 的双路扩展：主路和条件路各 20 个 DiT block，在 `[3,7,11,15,19]` 交流，前四次双向，最后一次条件路向主路传递。

本次实际训练：8 GPU、512×288、1000 optimizer steps、32,591 个训练图组。官方 RGB MSE + 0.3×截断 VGG-LPIPS；主路、条件路、交互模块及 VAE 按原规则更新，VAE time_conv 冻结。AdamW lr=2e-5，constant scheduler，BF16。历史只来自本模型此前生成结果的 uint8 往返，首帧 T1，后续 T5。

## 权重恢复

权重保存在本仓库私有 Release `bidirectional-step1000`，未将大文件加入普通 Git 历史。下载该 Release 的全部附件到 `assets/`，运行：

```bash
python3 scripts/restore_assets.py --assets assets
```

脚本先验证每份附件的 SHA256，再合并完整 checkpoint，并恢复基础模型和 LPIPS 权重。`weights/latest.pt` 是原完整 checkpoint：模型、optimizer、scheduler、每个 rank 的随机状态及数据顺序均保留。服务器原文件不改写。

基础初始化权重也保存在附件中，因为原模型构造器需要官方 Harmonizer、Cosmos DiT 和 tokenizer；这些权重的使用继续受上游许可约束。图像数据集不包含在仓库；附件只附训练/评测元数据清单。

## 环境

实际验证环境为 Linux、Python 3.10.20、PyTorch 2.7.1+cu126、torchvision 0.22.1、Transformer Engine 2.1.0，构建工具链 CUDA 12.8。完整版本见 `runtime/tanhe/requirements_resolved.txt`，已使用的上游运行源码在 `runtime/tanhe/vendor/`，无需依赖服务器之外的隐藏 Python 模块。

`bash scripts/install_environment.sh` 是安装入口。Transformer Engine 需要可用的 CUDA/cuDNN 开发环境；原服务器安装过程保存在 `scripts/install_tanhe_dependencies.sh`，其中机器路径及 GPU 架构必须按新硬件调整。

## 配置和推理

提供原图像数据，保持元数据中的原路径可访问（可使用挂载或符号链接）。当前数据默认位于 `/data/public_data/harmonizer_512_20261019`；`source=input.png`，监督是 `gt.png`，不是背景 `target.png`。

```bash
python3 scripts/configure.py --manifest data/manifests/sequences.jsonl
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/infer_bidirectional.py \
  --config configs/local.json --checkpoint weights/latest.pt \
  --manifest data/manifests/eval_5713_sequences.jsonl \
  --output outputs/bidirectional_eval
```

模型严格加载完整参数，六相机身份顺序固定，每条序列单独维护历史；序列边界沿用原清单，跨 context 的历史不会自动重置。推理输出 PNG、逐图清单、加载审计和状态。此入口不声称计算正式质量指标。

## 训练

```bash
export PYTHONPATH="$PWD:$PWD/runtime/tanhe/vendor"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=8 scripts/train_bidirectional.py \
  --config configs/local.json --output outputs/train_run --phase train --steps 1000
```

启动前先 `mkdir -p outputs/train_run`。训练器保留 Gloo 控制组、CPU staging、fsync 后原子保存的修复，避免 GPU 保存 checkpoint 时其他 rank 的 NCCL 超时。可通过 `--resume weights/latest.pt --steps 2000` 继续；必须维持 8 rank、相同清单字节、训练配方和模型结构，才能恢复各 rank 随机状态。新增运行会写到新目录。

可移植训练器与已完成运行实际训练器的唯一差异是所有者检查从服务器绝对路径改为仓库根目录。模型数学代码和训练循环不改动。历史 `implementation_sha256` 保留为原实验来源记录，新增的仓库文件校验由 `SOURCE_MANIFEST.json` 记录。

## 来源与验证范围

- `whole_data/runs/training_restart_20261024/bidirectional/`：原配置、完成状态和训练日志。
- `whole_data/runtime/training_restart_20261024/`：完成训练实际使用的修复版训练器及既有验证记录。
- `SOURCE_MANIFEST.json`：仓库源码快照校验。
- `release_manifest.json`：全部权重和附件的大小及 SHA256。
- `LICENSE`、`THIRD_PARTY_LICENSE.txt` 和 `SOURCE_PROVENANCE.json`：上游许可及来源；vendor 文件自带的版权说明保留。

打包时执行源码语法检查、模型文件与原实验哈希核对、运行入口检查及附件校验。此次打包不启动新的训练或占用 GPU，不能把打包检查当作移植后完成的 GPU 推理验证。`scripts/` 与 `configs/` 中其他历史入口保留作来源记录；本仓库推荐入口为上面的三个脚本。
