# tanhe-server 六相机与四卡验证

项目目录：`/data/juziwei/hlk/double_DiT`。本次仅做有界工程诊断，每项 4 个优化更新步；未启动正式训练、未保存训练检查点、未运行几何方案。原始模型、历史数据和第 1000 步权重保留。

## 数据范围

`data/tanhe_sixcam_validation/sequences.jsonl` 包含 4 条真实轨迹，每条 2 帧，每帧完整六相机。96 个输入/监督文件来自原机器可读源，逐文件核对原 source_index SHA256 后传输，并在目标服务器再次核对。没有复制相机来凑足六视角。

选用 train 划分的 scene-0021（21→32、63→70）、scene-0070（103→114）、scene-0074（22→26）。以相同 context 和资产身份关联时间轨迹；各帧来自原索引标记 ready 的完整组。相机内参、真实曝光时间和 ego pose 来自 nuScenes 官方标定记录，所有相机变换到该帧同一参考坐标系。

无可用插入资产的相机按索引使用渲染后的未编辑背景（源角色 target）作为输入；有插入资产的相机使用源角色 input。监督始终沿用原 DoubleDiT 配方的源角色 gt。这不是将角色 target 静默替换成监督，也没有用原始传感器图像代替缺失的渲染输入。

实际跨相机曝光跨度 40.555–42.606 ms。数据适合本次异步真实曝光的工程诊断；它不是严格同时曝光/同时渲染的六相机数据，不能据此宣布严格同步训练数据已就绪。

## 修复

首次六相机检查完成前向、历史反馈和四步梯度更新后，相机置乱检查失败。逐层探针确认：官方主路置乱差为 0，重复同输入的差为 0，几何路由一致，差异从首次跨相机注意力后出现。多个来源相机的 token 按输入 batch 顺序拼接，导致 BF16 注意力的归约顺序变化。

在 `double_dit/model_tanhe.py` 和 `double_dit/model_injection_tanhe.py` 中，按稳定语义相机 ID 排列已经由几何路由选中的来源 token。路由、来源集合、图像和输出顺序不变；测试阈值不变。修复前版本保存在 `backups/before_sixcam_fix_20261021/`。失败记录 `sixcam_v1_single` 和逐层探针记录保留。

## 四卡通信适配

NCCL 2.26.2 默认分配路径在初始化时报告 CUDA out-of-memory，当时每卡观测占用约 1 GiB；仅关闭 P2P 仍失败。使用 `--legacy-host-shm`，仅对子进程设置 `NCCL_CUMEM_HOST_ENABLE=0` 后通过初始化。该选项采用旧的 /dev/shm 路径，未修改系统配置或驱动，也没有关闭 P2P。NVIDIA 在 [NCCL 故障排查文档](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/troubleshooting.html) 中解释了旧版 cuMem host 分配对 NUMA 的依赖和此兼容选项。这里记录的是实测可用的兼容设置，未进一步断言服务器底层 NUMA 配置的具体根因。

## 验证协议

- 单卡：512×288、六相机，官方权重覆盖检查、T1/T5、生成历史反馈、四步梯度/参数更新、相机置乱等价、跨相机影响检查。
- 四卡：每 rank 都处理完整六相机轨迹，绝不把六相机拆成六个时间步。每步每 rank 使用两帧 T1→T5，四卡合计 48 个当前相机图像；共四个诊断更新步。
- 四卡各 rank 轮换真实轨迹，使用自己的生成输出构造下一帧历史。
- 每步检查所有实际参数梯度的 SHA256 在四个 rank 间完全一致；最后检查全模型参数 SHA256 一致，并核验主路首层和 VAE 编码器有实际参数更新。
- 记录每 rank 的 PyTorch peak allocated、peak reserved，以及周期采样的 nvidia-smi 设备占用。驱动层峰值是采样观测值，不能保证捕获任意短暂峰值。

## 已完成的结果（2026-10-21）

三种方案的单卡六相机和四卡 DDP 共 6 项真实模型诊断均通过。单卡相机置乱最大误差均为 0。四卡每个更新步的完整梯度哈希一致，结束时完整模型参数哈希一致；每个 rank 的官方 DiT 权重加载均覆盖 407 个张量、597,536,000 个元素（100%）。9 项 CPU 模块测试亦全部通过。

下面依次列出 PyTorch 分配峰值 / PyTorch 保留峰值 / 驱动层采样峰值，单位 GiB。四卡值取各 rank 或设备中的最大值，不是四卡相加：

- 单向读取：单卡六相机 **12.71 / 13.30 / 13.78**；四卡每卡 **16.54 / 17.35 / 17.96**。
- 联合注意力：单卡六相机 **12.46 / 13.14 / 13.63**；四卡每卡 **16.13 / 16.94 / 17.55**。
- 特征调制：单卡六相机 **12.69 / 13.35 / 13.83**；四卡每卡 **16.32 / 17.12 / 17.73**。

最终单卡目录为 `sixcam_v2_single`；最终四卡目录为 `sixcam_v4_ddp4`。完整数据、各 rank 数值、启动设置、权重加载覆盖率和源码哈希见 `reports/tanhe_sixcam_validation_summary.json`，底层显存采样见各目录的 `device_memory.json`。

原始 17 个文件哈希再次核验未变。此次诊断目录中未产生 `.pt` 检查点；原 `runs/train/latest.pt` 保留。所有诊断进程正常退出。失败尝试及日志也保留，未用成功结果覆盖失败证据。

## 运行入口

启动器 `scripts/run_sixcam_validation.py` 只接受 `single` 或 `ddp4`，没有正式训练选项。它检查 GPU 占用、获取 GPU 运行锁、拒绝覆盖已有目录，并记录实际代码/数据哈希。重复实验需使用新的 `--tag`。

```bash
python3 scripts/run_sixcam_validation.py --mode one_way --kind single --gpus 0 --tag sixcam_v2
python3 scripts/run_sixcam_validation.py --mode joint --kind single --gpus 1 --tag sixcam_v2
python3 scripts/run_sixcam_validation.py --mode modulation --kind single --gpus 2 --tag sixcam_v2

# 四卡诊断；共享同一组 GPU 的命令应等前一个退出再执行：
python3 scripts/run_sixcam_validation.py --mode one_way --kind ddp4 --gpus 0,1,2,3 --tag sixcam_v4 --legacy-host-shm
python3 scripts/run_sixcam_validation.py --mode joint --kind ddp4 --gpus 4,5,6,7 --tag sixcam_v4 --legacy-host-shm
python3 scripts/run_sixcam_validation.py --mode modulation --kind ddp4 --gpus 0,1,2,3 --tag sixcam_v4 --legacy-host-shm
```

配置、结果和显存记录保存于 `runs/tanhe_<mode>/<tag>_<kind>/`，日志为 `logs/<tag>_<mode>_<kind>.log`。单卡与 DDP 的工作负载不同，显存数字仅用于本次诊断容量检查，不应外推为正式长训练、1024×576 或质量排名。
