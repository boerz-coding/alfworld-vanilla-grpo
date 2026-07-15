# alfworld-vanilla-grpo

本仓库是一份可复现的实验记录:在 **ALFWorld(TextWorld 文字环境)** 上用 **vanilla GRPO** 训练 **Qwen2.5-7B-Instruct**,包含我们的训练配置、评测器和参考曲线(raw data)。

> **基座模型:`Qwen/Qwen2.5-7B-Instruct`(全参微调,非 LoRA)· 算法:GRPO · 环境:ALFWorld json_2.1.1 · 评测:valid_seen 140 局枚举,greedy**

## 结果

140 局固定评测集(枚举、greedy 解码)上的胜局数(/140)随训练步数,三次训练逐点列出。run_a 每 10 步有探针;run_b/run_c 当时只在 s10/s20/s140/s150 打了探针,故有空格:

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run_a(seed 0) | 10–11\* | — | 54 | 51 | 51 | 71 | 87 | 81 | 89 | 91 | 93 | 98 | 100 | 97 | 95 | 104 |
| run_b(seed 1) | 10–11\* | 48 | 51 | — | — | — | — | — | — | — | — | — | — | — | 85 | 78 |
| run_c(seed 2) | 10–11\* | 47 | — | — | — | — | — | — | — | — | — | — | — | — | 114 | 117 |

\* step-0 是未训练基座的静态测量(测过两次:10 与 11),三个 run 共享同一基座。

同一份数据的机器可读版:`reference/reference_curve.csv`;图:`reference/reference_curve.png`。

读数须知:同一 checkpoint 重复评测噪声约 ±2pp(实测三次复测差 ±4 局);不同训练 run 后半程散布较大(s150:78/104/117),曲线对照以**前半程(step 0–60)**为主。

## 评测方法

评测器是 `selfevolve/bootstrap.py`:枚举固定的 140 局 valid_seen 评测集(清单见 `reference/valid_seen_140_games.txt`),greedy 解码,输出逐局结果与总胜局数。它通过 **OpenAI 兼容 API**(`/v1` 端点)访问被评模型——我们的做法是用 vLLM serve checkpoint;任何能把模型暴露成 OpenAI 兼容推理端点的方式等效。

```bash
# 我们的评测命令(step-0 与逐 checkpoint 相同,换 --base-url / --out 即可):
python3 selfevolve/bootstrap.py --base-url http://127.0.0.1:8901/v1 \
  --enumerate --split eval_in_distribution --rollout-temp 0.0 --out out/probe_s<step>
# 完成标志:日志出现 COVERAGE_OK(140 局全部枚举)
```

评测器代码从产出参考曲线的仓库**原样隔离**(`MANIFEST.sha256` 覆盖全部 .py,`shasum -a 256 -c MANIFEST.sha256` 可校验)。曲线可比的前提是评测端一致——本仓库数字全部出自这一个评测器。

## 训练配置

- 超参冻结表:`spec/recipe.md`(verl 原始写法 + 各参数语义,便于翻译到其他训练框架)。
- prompt 模板:`agent_system/environments/prompts/alfworld.py`(与训练/评测同一份);逐字节协议与实测失配案例见 `spec/prompt_protocol.md`,训练期 dump 的逐字样本在 `reference/prompt_samples.txt`。
- **训练数据(容易漏,单独说)**:训练采样用的不是完整 train split,而是一个固定子集(2162/2435 任务目录,约 89%)。该子集源于我们另一组实验的特殊设计(预留了部分场景),对本复现而言只是固定的训练采样范围;参考曲线的三次训练均出自该子集。评测集不受影响:valid_seen 全量 140 局。训练前先构造(一次性):

  ```bash
  python3 selfevolve/make_scene_holdout.py \
    --data $ALFWORLD_DATA \
    --manifest selfevolve/scene_holdout_manifest.frozen.json \
    --out <路径>/alfworld_train_subset
  # 生成新数据目录:train/ 只含保留任务,valid_seen 等原样可用;
  # 此后训练与评测的 ALFWORLD_DATA 都指向这个新目录。
  ```

- 每 10 步保存一个 checkpoint,逐个评测得到曲线。

## 复现对照点(我们的经验记录)

1. **step-0**:冻结基座在本评测器下 = **10–11/140(7.1–7.9%)**。我们每次训练前都先跑一遍这个数;它不对,说明 prompt/版本/数据有失配,训练后的曲线也不会可比。
2. **前半程锚点**:见上表 step 10–20 列(s10:48/47;s20:54/51)——这是三个 run 之间一致性最好的区段。
3. 失配时的高频原因排序(我们踩过的):prompt 非逐字节一致 > 评测未走枚举/温度非 0 > 训练数据不一致 > `transformers`/`vllm` 版本漂移(我们钉在 4.51.3 / 0.8.5)> invalid-action 语义差异。细节见 `SOP.md` §7。

## 目录

| 路径 | 内容 |
|---|---|
| `SOP.md` | 我们端到端流程的逐步记录 + 失配排查经验 |
| `spec/recipe.md` | 训练超参冻结表 |
| `spec/prompt_protocol.md` | prompt 逐字节协议 |
| `agent_system/`, `selfevolve/` | 评测器及依赖(代码原样隔离,见 `MANIFEST.sha256`) |
| `reference/` | 参考曲线 CSV/PNG + prompt 逐字样本 + 140 局评测集清单 |
| `requirements-eval.txt` | 评测侧依赖(版本钉死项见注释) |

## 说明

- 代码与产出参考曲线的仓库逐字节一致,仅两个文件**注释**中的本地集群路径/机器名替换为占位符(功能零改动)。
- 逐局明细(每局 won/steps/actions 的 jsonl)未随仓库发布,需要时可提供。
- License:Apache 2.0(`LICENSE`、`Notice.txt`;环境与训练栈源自 verl-agent/GiGPO 生态)。
