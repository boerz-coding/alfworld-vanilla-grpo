# SOP — 从零到复现曲线

> 成本参考:我们在 4×H100 上一次 150 步训练 ≤24h;每个 checkpoint 的 140 局评测单卡 vLLM 约 20 分钟。

## 1. 环境

1. Python 3.12;`pip install -r requirements-eval.txt`。**`vllm==0.8.5`、`transformers==4.51.3` 请钉死**(transformers 升级实测会碰 Qwen2Tokenizer 兼容问题)。训练侧依赖由你们框架自定。
2. 下载 ALFWorld 数据 **json_2.1.1**(`alfworld-download` 或既有拷贝),`export ALFWORLD_DATA=<数据根>`。

## 2. 训练数据子集(为了曲线可比,必须一致)

参考曲线的训练采样用的是一个**固定的 train 子集**,用仓库内脚本 + 冻结清单构造:

```bash
python3 selfevolve/make_scene_holdout.py \
  --data $ALFWORLD_DATA \
  --manifest selfevolve/scene_holdout_manifest.frozen.json \
  --out <路径>/alfworld_train_subset
```

训练 rollout 从这棵树的 train split 采样。清单文件是冻结的,不要重新生成。

## 3. Step-0 校验(硬闸门,开训前必须过)

**目的:证明"你们 serve 的模型 + 本仓库评测器"= 我们当初的读数;不过则后面所有数字无意义。**

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8901
python3 selfevolve/bootstrap.py \
  --base-url http://127.0.0.1:8901/v1 \
  --enumerate --split eval_in_distribution --rollout-temp 0.0 \
  --out out/step0
```

**通过判据**:日志出现 `COVERAGE_OK`(140 局全部枚举完成);胜局 **10–11/140**。
不过 → 不要开训,按 §7 排查(九成是 prompt/版本问题)。

## 4. 训练(你们的框架)

- 超参逐行对照 `spec/recipe.md`(语义栏是给非 verl 框架的翻译)。
- **prompt 必须逐字节一致**:优先直接 `import` 本仓库 `agent_system/environments/prompts/alfworld` 的模板;自行实现则按 `spec/prompt_protocol.md` 对拍。
- **每 10 步存一个 HF 可加载的 checkpoint**(评测要逐个 serve)。
- 建议记录的训练遥测:每步 invalid-action 率。参考:step-0 为 0.575/步,**训练后 20 步内应降到 ≈0**——这是最灵敏的早期对齐信号。

## 5. 逐 checkpoint 评测

对每个 checkpoint(s10, s20, …, s150):

```bash
vllm serve <ckpt_path> --port 8901
python3 selfevolve/bootstrap.py \
  --base-url http://127.0.0.1:8901/v1 \
  --enumerate --split eval_in_distribution --rollout-temp 0.0 \
  --out out/probe_s<step>
```

- 每次评测必须出现 `COVERAGE_OK`;产物 `games_round1.jsonl`(140 行逐局结果)**全部保留**。

## 6. 对表

1. **曲线层**:step→wins/140 对照 `reference/reference_curve.csv`。对齐目标按强度排序:
   - step-0 = 10–11/140(硬);
   - s10–s20 落在 45–57/140 附近(我们 3 次训练:47/48/51/54);
   - s60 前后进入 55–65% 量级(参考 run_a:87/140);
   - s150 参考散布 [78, 117]/140,**单点不作成败判据**。
2. **逐局层(可选,更强)**:`python3 tools/mcnemar.py out/probe_s60/games_round1.jsonl reference/games/ref_run_a_s60.jsonl`,同步位 p>0.05 即逐局一致。
3. 汇报最小包:step→wins CSV + 全部 games jsonl + 你们配置相对 `spec/recipe.md` 的逐行 diff(一致/等效/做不到 三栏)。

## 7. 对不上时的排查顺序(按命中率)

1. **prompt 不逐字节一致**(历史窗口、admissible actions 列表格式、`<think>/<action>` 说明、空格换行)→ 与 `reference/prompt_samples.txt` diff。
2. 评测没走枚举/温度不为 0 → 确认命令行 + `COVERAGE_OK`。
3. 数据不是 json_2.1.1,或训练没用 §2 的固定子集。
4. `transformers`/`vllm` 版本漂移(tokenizer 行为变化)。
5. invalid action 的判定与惩罚语义不同(见 recipe 表对应行)。
6. 以上排除仍差 → 把你们的 games jsonl + 训练遥测发给我们,逐局对差异局。

## 8. 噪声底

- 同 checkpoint 重复评测:±2pp(我们实测重复三次差 ±4 局)。
- 单点差 <4pp 不构成结论;曲线形状(step-0 → 前 60 步爬坡)比任何单点都重要。
