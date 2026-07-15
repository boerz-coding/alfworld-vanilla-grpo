# alfworld-vanilla-grpo

在 **ALFWorld(TextWorld 文字环境)** 上用 **vanilla GRPO** 训练 **Qwen2.5-7B-Instruct** 的完整工程包:训练配方、评测工具、参考曲线(raw data)。感谢您参与这次跨框架复现,任何一步卡住都欢迎随时联系我们对齐。

**分工只有一条**:训练用贵侧自己的 RL 框架(照 `spec/recipe.md` 对齐超参);评测请直接用本仓库自带的评测器(`selfevolve/bootstrap.py`,只需把 checkpoint 用 vLLM 起成 OpenAI 兼容端口)。评测器代码是从产出参考曲线的仓库**原样隔离**出来的,`MANIFEST.sha256` 覆盖全部 .py 文件、可逐一校验(`shasum -a 256 -c MANIFEST.sha256`)——评测这把"尺子"两边一致,曲线才可比,恳请不要另行实现。

## 目标曲线

140 局固定评测集(枚举、greedy 解码)上的成功率随训练步数:

| step | 0 | 20 | 40 | 60 | 100 | 150 |
|---|---|---|---|---|---|---|
| wins/140 | **10–11** | 51–54 | 51 | 87 | 93 | 78–117(3 次训练的散布) |

raw data 在 `reference/reference_curve.csv`(3 次训练全部点位),图在 `reference/reference_curve.png`。两个读数须知:同一 checkpoint 重复评测噪声约 ±2pp;不同训练 run 后半程散布较大,**前半程(step 0–60)是主要对齐目标**。逐局明细如有需要,联系我们另行提供。

## 复现流程(总览;每步细节见 `SOP.md`)

1. **装环境**:`pip install -r requirements-eval.txt`(**`vllm==0.8.5`、`transformers==4.51.3` 请钉死**,升级实测会碰 tokenizer 兼容问题);下载 ALFWorld 数据 **json_2.1.1**,设 `ALFWORLD_DATA`。
2. **造训练子集**:`python3 selfevolve/make_scene_holdout.py --data $ALFWORLD_DATA --manifest selfevolve/scene_holdout_manifest.frozen.json --out <路径>`——参考曲线的训练采样用的就是这个固定子集,清单已冻结,直接用即可。
3. **Step-0 校验(请务必先过)**:vLLM 起裸基座,跑一次评测(命令见下),预期 `COVERAGE_OK` + **10–11/140**。这一步证明"贵侧 serve 的模型 + 本评测器 = 我们当初的读数";若不过,请先按 `SOP.md` §7 排查(九成是 prompt 或版本问题),暂缓开训。
4. **训练**:贵侧框架 + `spec/recipe.md` 超参表;prompt 必须逐字节一致(优先直接 import 本仓库 `agent_system/environments/prompts/alfworld` 的模板,或按 `spec/prompt_protocol.md` 对拍);**每 10 步存一个 HF 可加载 checkpoint**。
5. **逐 checkpoint 评测**:对 s10…s150 逐个 serve 并跑同一条评测命令,每次须见 `COVERAGE_OK`;产物请全部保留。
6. **对表**:step→wins 对照 `reference/reference_curve.csv`;判读顺序:step-0(硬)→ s10–20 落在 45–57/140 → s60 进入 55–65% 量级 → s150 参考散布 [78,117]/140(单点不作成败判据)。

评测命令(step-0 与逐 checkpoint 通用,换 `--out` 即可):

```bash
vllm serve <模型或checkpoint路径> --port 8901
python3 selfevolve/bootstrap.py --base-url http://127.0.0.1:8901/v1 \
  --enumerate --split eval_in_distribution --rollout-temp 0.0 --out out/probe_s<step>
```

## 目录

| 路径 | 内容 |
|---|---|
| `SOP.md` | 上述流程的逐步展开 + 排查手册 |
| `spec/recipe.md` | 训练超参冻结表(verl 原始写法 + 语义解释,便于翻译到贵侧框架) |
| `spec/prompt_protocol.md` | prompt 逐字节协议(最常见的失配源) |
| `agent_system/`, `selfevolve/` | 评测器及依赖(代码原样隔离,见 `MANIFEST.sha256`) |
| `reference/` | 参考曲线 CSV/PNG + prompt 逐字样本 + 140 局评测集清单 |
| `requirements-eval.txt` | 评测侧依赖 |

## 说明

- 代码与产出参考曲线的仓库逐字节一致,仅两个文件的**注释**中的本地集群路径/机器名替换为占位符(功能零改动);`scene_holdout_manifest.frozen.json` 的路径字段同样已规范化,功能字段未动。`MANIFEST.sha256` 以本仓库文件为准,供传输完整性校验。
- License:Apache 2.0(`LICENSE`、`Notice.txt`;环境与训练栈源自 verl-agent/GiGPO 生态)。
