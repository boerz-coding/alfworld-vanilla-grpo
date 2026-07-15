# 训练超参冻结表 — vanilla GRPO(与参考曲线的训练完全一致)

> "verl 参数"是我们框架的原始写法;"语义"栏解释各参数含义,便于翻译到其他训练框架。**语义等效优先于名字相同**。

| 项 | verl 参数 | 值 | 语义 |
|---|---|---|---|
| 基座 | `actor_rollout_ref.model.path` | `Qwen/Qwen2.5-7B-Instruct` | 全参微调(非 LoRA) |
| 算法 | `algorithm.adv_estimator` | `grpo` | 组内均值基线 advantage;全成/全败组 advantage=0 |
| 组大小 | `env.rollout.n` | **8** | 同一任务采 8 条轨迹为一组 |
| 每步任务数 | `data.train_batch_size` | **16** | 16 任务 × 8 rollouts = **128 局/步**,on-policy |
| 学习率 | `actor.optim.lr` | **1e-6** | 常数,无 warmup/decay(AdamW,verl 默认) |
| mini-batch | `actor.ppo_mini_batch_size` | 256 | 每步样本一次梯度扫过(等效 1 epoch) |
| micro-batch | `ppo_micro_batch_size_per_gpu` | 16 | 纯显存切分,不影响梯度 |
| KL | `use_kl_loss=True, kl_loss_coef=0.01, kl_loss_type=low_var_kl` | 0.01 | KL 作 loss 项,参考策略=初始模型;**reward 不加 KL**(`use_kl_in_reward=False`) |
| 非法动作惩罚 | `use_invalid_action_penalty=True, coef=0.1` | 0.1 | 非法动作在 loss 侧惩罚(verl-agent 语义) |
| 奖励 | — | 二值 | 局终 won=1 / 否则 0,无 shaping |
| prompt 预算 | `data.max_prompt_length` | 2048 | 超长直接报错(`truncation='error'`),不静默截断 |
| response 预算 | `data.max_response_length` | 512 | 每步生成上限 |
| 训练采样温度 | rollout 默认 | 未显式设置(verl 默认) | 评测温度与此无关(评测器固定 0.0) |
| 单局步数上限 | `env.max_steps` | **50** | 50 步未完成判负 |
| 环境 | `env.env_name` | `alfworld/AlfredTWEnv` | TextWorld 文字版 ALFRED |
| 训练数据 | `ALFWORLD_DATA` | SOP §2 的固定 train 子集 | 清单冻结,保证与参考曲线可比 |
| 总步数 | `trainer.total_epochs` | **150** | 每步新采样 |
| 存档频率 | `trainer.save_freq` | **10** | 每 10 步一个 checkpoint |
| env seed | `env.seed` | 0 / 1 / 2 | 参考曲线三次训练分别用的种子 |
| 硬件(参考) | — | 4×H100,rollout TP=2 | 不要求一致,吞吐不影响动力学 |

## 备注

- 训练中框架自带的 validation(我们设 `test_freq=10`,val 温度 0.4)只是遥测,**不进对表**;正式曲线一律来自 `selfevolve/bootstrap.py` 的枚举评测(SOP §5)。
- 本表与产出参考曲线的训练脚本逐行核对过。复现框架若存在语义对不上的项,记录成 diff 清单便于双方对账;评测侧保持一致是曲线可比的前提。
