# 实跑命令(verl-agent 栈,原文)

是的,`spec/recipe.md` 表中就是 **verl 参数**——我们的训练栈是 verl-agent(GiGPO)生态。本仓库不含训练代码(只含评测器);下面是产出参考曲线的完整实跑命令,变量已按实跑值展开,路径换成占位符。

## 数据准备(一次性)

```bash
python3 -m examples.data_preprocess.prepare \
    --mode 'text' --local_dir $DATA_DIR \
    --train_data_size 16 --val_data_size 128
```

## 训练(三次参考训练共用,仅 env.seed 不同:0 / 1 / 2)

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/text/train.parquet \
    data.val_files=$DATA_DIR/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    "trainer.logger=['console','tensorboard']" \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=grpo_7b_none \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.resume_mode=auto \
    trainer.default_local_dir=$CKPT_DIR
```

环境变量:`ALFWORLD_DATA` 指向 SOP §2 构造出的数据目录;`HF_HOME` 按各自环境;离线集群上我们另设 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline`。

## 训练遥测(wandb 未用,tensorboard 全量记录)

日志器为 `['console','tensorboard']`(集群无外网,wandb 走不了在线);wandb 会记的量 TensorBoard 全有(84 个 scalar tag)。三次参考训练的关键遥测已导出:

- `reference/telemetry/run_{a,b,c}_telemetry.csv` — 逐步:train rollout success、val success(temp 0.4)、reward、kl_loss、entropy、pg_loss、grad_norm、valid_action_ratio、episode/response 长度;
- `reference/telemetry/telemetry_overview.png` — 三 run 总览(success / KL·entropy / valid_action_ratio)。

注意:训练中的 `val/success_rate` 用 temp 0.4 抽样,是遥测;正式曲线一律来自 README「评测方法」的枚举评测器(greedy),两者数值不同属预期。TensorBoard 原始 event 文件需要的话可另行提供。
