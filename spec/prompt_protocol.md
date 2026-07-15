# Prompt 模板逐字节协议

**为什么苛刻到字节**:prompt 形态是这套复现里最大的静默失配源——格式差异会同时移动成功率和非法动作率,整条曲线平移且没有任何报错。SOP §3 的 step-0 校验就是为了在烧卡之前抓住它。

## 模板结构(两种)

模板真身:`agent_system/environments/prompts/alfworld.py`(本仓库自带,与产出参考曲线的代码逐字节一致,见 MANIFEST)。逐字样本:`reference/prompt_samples.txt`(训练时通过 `ALFWORLD_PROMPT_DUMP` 环境变量直接 dump 的原件)。

1. **首步(init=True)**:身份句("You are an expert agent operating in the ALFRED Embodied Environment.")+ 当前观察(含 TextWorld 欢迎行)+ 任务句 + **admissible actions 列表**(注意:numpy 风格的方括号多行列表,元素带引号、换行分隔无逗号——这个怪格式也要保留)+ 行动指示(`<think></think>` 内推理、`<action></action>` 内输出动作)。
2. **续步(init=False)**:身份句 + 任务句 + "Prior to this step, you have already taken N step(s). Below are the most recent K observations and the corresponding actions you took:" + 历史窗口 + 当前 admissible actions + 同样的行动指示。**历史窗口大小以 prompts 模块代码为准,不要猜**。

## 自检方法(二选一,推荐 A)

- **A(零风险)**:训练 harness 直接 `from agent_system.environments.prompts.alfworld import ...` 复用本仓库的模板/构造逻辑。
- **B(自行实现)**:训练首步把贵侧拼出的 prompt dump 下来,与 `reference/prompt_samples.txt` **逐字符 diff**(空格、换行、引号、方括号都算),diff 干净再开训。

## 生成侧约定

- 动作解析:取 `<action></action>` 内文本作为环境动作;非法/不可解析按 recipe 表的 invalid-penalty 语义处理;评测器自行统计 `n_invalid`,无需额外干预。
- Chat template:走 Qwen2.5-Instruct tokenizer 自带的官方 chat template,不要手拼 system/user 包装——这也是钉死 `transformers==4.51.3` 的原因之一。
