# alfworld-vanilla-grpo

在 **ALFWorld(TextWorld 文字环境)** 上用 **vanilla GRPO** 训练 **Qwen2.5-7B-Instruct** 的完整工程包:训练配方、评测工具、参考曲线(含逐局 raw data)。

- **训练**:用你们自己的 RL 框架,按 `spec/recipe.md` 的超参表对齐。
- **评测**:用本仓库自带的 `selfevolve/bootstrap.py`(把 checkpoint 用 vLLM 起成 OpenAI 兼容端口即可)。评测器代码是从我们产出参考曲线的仓库**原样隔离**出来的,`MANIFEST.sha256` 可逐文件校验(`shasum -a 256 -c MANIFEST.sha256`)。**请不要自写评测**——尺子不同,曲线没法比。

## 目标曲线

140 局固定评测集(枚举、greedy)上的成功率随训练步数:

| step | 0 | 20 | 40 | 60 | 100 | 150 |
|---|---|---|---|---|---|---|
| wins/140 | **10–11** | 51–54 | 51 | 87 | 93 | 78–117(3 次训练的散布) |

完整 raw data:`reference/reference_curve.csv`(3 次训练全部点位)、`reference/games/`(逐局结果 jsonl)、`reference/reference_curve.png`(图)。

同一 checkpoint 重复评测的噪声约 ±2pp;不同训练 run 之间后半程散布较大(见图),**前半程(step 0–60)是主要的对齐目标**。

## 目录

| 路径 | 内容 |
|---|---|
| `SOP.md` | 从零到出曲线的步骤(含 step-0 校验闸门) |
| `spec/recipe.md` | 训练超参冻结表(附 verl 原始写法与语义解释) |
| `spec/prompt_protocol.md` | prompt 模板逐字节协议(最常见的失配源) |
| `agent_system/`, `selfevolve/` | 评测器及其依赖(原样隔离,见 MANIFEST) |
| `reference/` | 参考曲线 raw data + prompt 逐字样本 + 评测局清单 |
| `tools/mcnemar.py` | 逐局配对显著性对比(可选) |
| `requirements-eval.txt` | 评测侧依赖(版本钉死项见注释) |

## 快速开始

```bash
pip install -r requirements-eval.txt          # 注意 vllm/transformers 版本钉死
export ALFWORLD_DATA=<你的 alfworld 数据根>    # json_2.1.1
# step-0 校验(必须先过,详见 SOP §3):
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8901 &
python3 selfevolve/bootstrap.py --base-url http://127.0.0.1:8901/v1 \
  --enumerate --split eval_in_distribution --rollout-temp 0.0 --out out/step0
# 预期:输出 COVERAGE_OK,10–11/140
```
