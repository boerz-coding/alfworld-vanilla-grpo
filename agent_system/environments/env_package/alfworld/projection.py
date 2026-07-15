# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
import re

def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-30:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            actions[i] = actions[i][-30:]

        # NOTE (2026-07-10): the <think>...</think> requirement was REMOVED for the
        # frozen-policy self-evolve harness, for the SAME reason as the Chinese-CoT
        # removal below. It forced valids=0 whenever the completion lacked a LITERAL
        # <think></think> pair -- even when a well-formed, admissible <action> was
        # extracted and executed. This catastrophically mis-measured reasoning models
        # served with a vLLM reasoning parser (Qwen3.5-4B): the parser routes the
        # thinking into `reasoning_content`, so the `content` the harness reads has a
        # perfect <action> but NO literal <think> tags -> EVERY step flagged invalid.
        # Signature that exposed it: 31/40 WON 4B games had "every step invalid", a
        # logical impossibility (you cannot win ALFWorld without valid actions).
        # inv/step for the 4B was ~0.985 (artifact) vs 0.21/0.58 for 8B/7B. Only the
        # COUNTER was wrong -- `action` is executed regardless of `valids`, so SUCCESS
        # rates were never affected; only n_invalid / inv/step were. We need the action,
        # not proof the model emitted think-tags in-band. (The think-format constraint
        # belongs to SkillRL's RL training, whose own projection.py copy is untouched.)

        # NOTE (2026-07-09): the Chinese-character rejection was REMOVED for the
        # frozen-policy self-evolve harness. It marked an otherwise-valid action
        # invalid whenever the model REASONED in Chinese (Qwen2.5-1.5B does this by
        # default; 7B sometimes) even though the extracted <action> is a correct
        # English/admissible command. We only need the action; the CoT language is
        # irrelevant here. This artifact was silently confounding the 1.5B/7B
        # baselines (n_invalid ~23/44 of 50 -> near-0% success) \u2014 a parsing tax,
        # not a capability gap. (The English-CoT constraint belongs to SkillRL's RL
        # training, whose own projection.py copy is untouched.)

    return actions, valids
