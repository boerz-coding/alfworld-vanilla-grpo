# 流程记录 — 我们如何从零跑出参考曲线

> 成本参考:150 步训练在 4×H100 上 ≤24h;每个 checkpoint 的 140 局评测,单卡 vLLM 约 20 分钟。

## 1. 环境

- Python 3.12;评测侧依赖见 `requirements-eval.txt`。其中 **`vllm==0.8.5`、`transformers==4.51.3` 是钉死的**——transformers 升级实测碰到过 Qwen2Tokenizer 兼容问题。
- ALFWorld 数据 **json_2.1.1**,`ALFWORLD_DATA` 指向数据根。

## 2. 训练数据

训练采样使用 train split 的一个固定子集(2162/2435 任务目录,约 89%;评测集 valid_seen 不受影响,全量 140 局),三次参考训练均在该子集上。构造方式:

```bash
python3 selfevolve/make_scene_holdout.py \
  --data $ALFWORLD_DATA \
  --manifest selfevolve/scene_holdout_manifest.frozen.json \
  --out <路径>/alfworld_train_subset
```

清单文件是冻结版本,直接使用即可。

构造出的子集树对 valid_seen 等其余 split 做了软链,因此**训练与评测可将 `ALFWORLD_DATA` 统一指向子集树**——我们即如此运行(评测集不受裁剪影响)。

## 3. Step-0 检查(我们每次训练前都先跑)

用评测器测一遍**未训练的基座**,确认整条链路(模型服务 + prompt + 数据 + 评测器)与参考读数一致:

```bash
# 我们的做法:vLLM 起基座;任何 OpenAI 兼容端点等效
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8901
python3 selfevolve/bootstrap.py \
  --base-url http://127.0.0.1:8901/v1 \
  --enumerate --split eval_in_distribution --rollout-temp 0.0 \
  --out out/step0
```

参考读数:`COVERAGE_OK` + **10–11/140**。这个数不对时,我们的经验是先修链路再训练——训练后的曲线差异会与链路失配混在一起,无法归因。

## 4. 训练

- 超参:`spec/recipe.md` 冻结表,与三次参考训练逐行一致。
- prompt:训练与评测用同一份模板(`agent_system/environments/prompts/alfworld.py`);跨框架复现时模板逐字节一致是曲线可比的前提,对拍方法见 `spec/prompt_protocol.md`。
- checkpoint:每 10 步保存一个(HF 可加载格式,评测要逐个 serve)。

## 5. 逐 checkpoint 评测

对 s10, s20, …, s150 逐个执行与 §3 相同的评测命令(换模型端点与 `--out`)。每次评测以 `COVERAGE_OK` 为完成标志;产物 `games_round1.jsonl`(140 行逐局结果)我们全部留档——逐局是对账的最小单位。

## 6. 对表

step→wins/140 与 `reference/reference_curve.csv` 对照。我们建议的判读顺序(按信号强度):

1. step-0 = 10–11/140;
2. s10–s20 落在 45–57/140 附近(参考三跑:47/48/51/54);
3. s60 前后进入 55–65% 量级(run_a 参考:87/140);
4. s150 的参考散布是 [78, 117]/140——后半程 run 间方差大,单点不适合作成败判据。

## 7. 失配排查经验(按我们实际踩坑的频率排序)

1. **prompt 非逐字节一致**(历史窗口、admissible actions 列表格式、`<think>/<action>` 说明、空格换行)→ 与 `reference/prompt_samples.txt` 逐字符 diff。
2. 评测未走枚举 / 温度不为 0(抽样≠枚举)→ 核对命令行与 `COVERAGE_OK`。
3. 数据不是 json_2.1.1,或训练数据不是 §2 的固定子集。
4. `transformers`/`vllm` 版本漂移(tokenizer 行为变化)。
5. invalid action 的判定与惩罚语义差异(见 recipe 表对应行)。
6. 以上全排除仍差:双方交换逐局 jsonl 与训练遥测,按局对差异,通常几局就能定位。

## 8. 噪声底

- 同 checkpoint 重复评测:±2pp(实测三次复测差 ±4 局)。
- 单点差 <4pp 不构成结论;曲线形状(step-0 → 前 60 步爬坡)比任何单点都可靠。
