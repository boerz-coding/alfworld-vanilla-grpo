"""
Phase-0 frozen-weights skill self-evolution harness for ALFWorld.

A FROZEN policy model (served by vLLM through an OpenAI-compatible API) plays
ALFWorld games with the current skill library injected into its prompt, writes
candidate skills from its own failed/successful trajectories (same endpoint,
same weights), and a paired-comparison GATE decides whether the candidate
batch enters the library. Upstream SkillRL accepts generated skills
unconditionally; the gate (and its accept_all / random_matched controls) is
the deliberate contribution of this harness and a planned ablation axis.

The library lives in SkillRL's exact claude_style_skills.json container
(top-level keys: general_skills / task_specific_skills / common_mistakes /
metadata), so examples/grpo_trainer/run_alfworld_skills*.sh can consume any
snapshot produced here unchanged.

No ray, no verl: a plain thread pool over one OpenAI client plus direct
AlfredTWEnv instances, mirroring the prompt/action protocol of
agent_system/environments/env_manager.py (AlfWorldEnvironmentManager) and
agent_system/multi_turn_rollout/rollout_loop.py (one single-user-message chat
completion per env step).

Example (against any OpenAI-compatible endpoint):
    python selfevolve/bootstrap.py \
        --base-url http://127.0.0.1:8901/v1 \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --smoke
"""

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# textworld's PDDL grammar parser is a module-global tatsu instance and is NOT
# thread-safe: concurrent load/reset AND step (textgen grammar) corrupt its rule stack
# ("IndexError: pop from empty list" observed in a smoke run).
# ALL textworld interactions (construct/reset/step) must hold this lock.
ENV_LOAD_LOCK = threading.Lock()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ALF_CONFIG_PATH = os.path.join(
    REPO_ROOT,
    "agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
)

# ---------------------------------------------------------------------- #
# Heavy repo/runtime imports are guarded so that `--help` (and py_compile)
# works on machines without alfworld / openai / torch installed. Actual
# runs must happen where the full stack is installed.
# ---------------------------------------------------------------------- #
try:
    from openai import OpenAI
    from agent_system.memory.skills_only_memory import SkillsOnlyMemory
    from agent_system.environments.prompts.alfworld import (
        ALFWORLD_TEMPLATE,
        ALFWORLD_TEMPLATE_NO_HIS,
        ALFWORLD_TEMPLATE_WITH_MEMORY,
    )
    from agent_system.environments.env_package.alfworld.projection import (
        alfworld_projection,
    )
    from agent_system.environments.env_package.alfworld.envs import (
        load_config_file,
    )
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment import (
        get_environment,
    )
    _IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - record and defer to runtime
    _IMPORT_ERROR = e


# Task categories: same six keys as memory_data/alfworld/claude_style_skills.json
# and skill_generation/alfworld.py.
TASK_TYPES = [
    'pick_and_place',
    'look_at_obj_in_light',
    'clean',
    'heat',
    'cool',
    'examine',
]

# Copied from skill_generation/alfworld.py::generate_task_specific_skills
TASK_DESCRIPTIONS = {
    'pick_and_place': 'Pick up object(s) from one location and place them at a target location',
    'look_at_obj_in_light': 'Find an object and examine it under a light source (usually desklamp)',
    'clean': 'Find an object, clean it in a sink/basin, then place it somewhere',
    'heat': 'Find an object, heat it in microwave, then place it somewhere',
    'cool': 'Find an object, cool it in fridge, then place it somewhere',
    'examine': 'Find and examine a specific object',
}

# Offset added to --seed to derive the FIXED gate-game seed. The gate set is
# drawn from --split (train by default; the self-evolution protocol never
# gates on valid_seen/valid_unseen) and is identical for every round and for
# both gate conditions.
GATE_SEED_OFFSET = 777000


# ---------------------------------------------------------------------- #
# Stratified gate window + scoring, ported from Boer's original harness
# (GenericAgent_RL_logits/src/alfworld_env.py:156-210). Gamefile paths embed
# the ALFWorld task-type directory prefix, so a game's type is known WITHOUT
# resetting it — the gate window can be type-balanced up front.
# ---------------------------------------------------------------------- #

TASK_TYPE_PREFIXES = (
    "pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
)

# Library task type -> gamefile prefixes that (predominantly) generate it,
# used to CONCENTRATE the per-type gate window on candidate types. Not a
# bijection: goal-text detection is what finally buckets a record (an
# 'examine' record comes from "examine the X with the desklamp" goals of
# look_at games; pick_two goals detect as pick_and_place), so per-type
# verdicts still group by each record's detected type — this map only
# decides which games are WORTH playing.
LIB_TYPE_TO_PREFIXES = {
    'pick_and_place': ('pick_and_place_simple', 'pick_two_obj_and_place'),
    'look_at_obj_in_light': ('look_at_obj_in_light',),
    'examine': ('look_at_obj_in_light',),
    'clean': ('pick_clean_then_place_in_recep',),
    'heat': ('pick_heat_then_place_in_recep',),
    'cool': ('pick_cool_then_place_in_recep',),
}


def doc_map_key(gamefile):
    """Root-independent '<taskdir>/<trial>' key of a gamefile path, shared with
    analysis/make_oracle_location_docs.py so --doc-map-file lookups survive a
    move of ALFWORLD_DATA."""
    if not gamefile:
        return None
    parts = os.path.normpath(gamefile).split(os.sep)
    return '/'.join(parts[-3:-1]) if len(parts) >= 3 else None


def game_type(p):
    """Task-type prefix of a gamefile path (alfworld_env.py:161-165)."""
    for seg in (p or '').replace("\\", "/").split("/"):
        if seg.split("-")[0] in TASK_TYPE_PREFIXES:
            return seg.split("-")[0]
    return "?"


def native_props(all_games):
    """Native task-type proportions of a game pool (alfworld_env.py:167-174)."""
    counts = {t: 0 for t in TASK_TYPE_PREFIXES}
    for g in all_games:
        t = game_type(g)
        if t in counts:
            counts[t] += 1
    total = sum(counts.values()) or 1
    return {t: counts[t] / total for t in TASK_TYPE_PREFIXES}, counts


def stratified_games(all_games, seed, n):
    """
    Deterministic type-balanced window of ~n games (largest-remainder
    proportional to the pool's type mix). Returns (games, native_props).
    Each present type gets >=1 when n >= #present-types.

    Port of alfworld_env.py:177-199 (random.Random shuffle instead of numpy
    RandomState permutation; same per-type-seed determinism guarantee).
    """
    props, counts = native_props(all_games)
    by = {t: [g for g in all_games if game_type(g) == t] for t in TASK_TYPE_PREFIXES}
    present = [t for t in TASK_TYPE_PREFIXES if counts[t] > 0]
    # largest-remainder allocation summing to n, floor 1 per present type
    raw = {t: n * counts[t] / sum(counts[t] for t in present) for t in present}
    alloc = {t: max(1, int(raw[t])) for t in present}
    # trim/grow to hit exactly n
    while sum(alloc.values()) > n:
        alloc[max(alloc, key=lambda t: alloc[t])] -= 1
    for t in sorted(present, key=lambda t: raw[t] - int(raw[t]), reverse=True):
        if sum(alloc.values()) >= n:
            break
        alloc[t] += 1
    games = []
    for t in present:
        k = min(alloc[t], len(by[t]))
        indices = list(range(len(by[t])))
        random.Random(seed + TASK_TYPE_PREFIXES.index(t)).shuffle(indices)
        games.extend(by[t][i] for i in indices[:k])
    return games, props


def proportional_stratified_mean(records, props):
    """
    Per-type success mean weighted by NATIVE type proportions, so the
    dominant type (pick_and_place) cannot mask another type's regression;
    invariant to which types were sampled. Port of alfworld_env.py:202-210
    (proportional_mean), taking each record's type from its gamefile.
    """
    per = {t: [] for t in TASK_TYPE_PREFIXES}
    for rec in records:
        t = game_type(rec['gamefile'])
        if t in per:
            per[t].append(float(rec['won']))
    hit = [t for t in TASK_TYPE_PREFIXES if per[t]]
    z = sum(props[t] for t in hit) or 1.0
    return sum(props[t] * (sum(per[t]) / len(per[t])) for t in hit) / z


def detect_task_type(task_description):
    """
    Keyword task-type detection, replicated from
    SkillsOnlyMemory._detect_task_type (ALFWorld branch) so it also works
    while the library is still empty.
    """
    goal = task_description.lower()
    if 'look at' in goal and 'under' in goal:
        return 'look_at_obj_in_light'
    elif 'clean' in goal:
        return 'clean'
    elif 'heat' in goal or 'hot' in goal:  # 'hot'-phrased heat goals (offline: 10/16 heat misrouted to pick)
        return 'heat'
    elif 'cool' in goal:
        return 'cool'
    elif 'examine' in goal:  # drop 'find' — it steals pick_two "find two..." goals (offline: 8/24)
        return 'examine'
    else:
        return 'pick_and_place'


def empty_library():
    """Empty skill bank in the exact claude_style_skills.json container."""
    return {
        'general_skills': [],
        'task_specific_skills': {t: [] for t in TASK_TYPES},
        'common_mistakes': [],
        'metadata': {
            'source': 'self-evolved from ALFWorld trajectories by the frozen policy '
                      '(selfevolve/bootstrap.py)',
        },
    }


# ---------------------------------------------------------------------- #
# Frozen-policy client
# ---------------------------------------------------------------------- #

class ChatClient:
    """One OpenAI-compatible endpoint used for BOTH acting and skill writing."""

    def __init__(self, base_url, model, api_key='dummy', timeout=300.0, max_retries=3):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.max_retries = max_retries

    def __call__(self, prompt, temperature, max_tokens, system=None, seed=None):
        # ACTOR calls pass system=None -> messages=[{user}], BYTE-IDENTICAL to
        # the original single-user-message step format (rollout_loop.py
        # preprocess_single_sample). Only the whole-doc WRITER (--skill-style
        # doc) passes a system prompt, making its call [{system}, {user}]; the
        # itemized writer and every actor step stay single-user.
        messages = ([{"role": "system", "content": system}] if system else []) \
            + [{"role": "user", "content": prompt}]
        body = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # vLLM's OpenAI-compatible chat/completions honors "seed" for reproducible
        # sampling. Only the doc best-of-N writer passes a seed; every other caller
        # (actor steps, itemized writer) leaves seed=None -> field omitted -> the
        # request body is byte-identical to before this change.
        if seed is not None:
            body["seed"] = seed
        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**body)
                return response.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - transport errors, retry
                last_err = e
                time.sleep(2 ** attempt)
        detail = str(last_err)
        if "context length" in detail.lower() or "maximum context" in detail.lower():
            # Distinct, greppable marker: this is a config problem (prompt > server
            # context), not a transient transport error. Silent 400s here caused the
            # doc arm to no-op for 12 rounds (bug found 2026-07-08).
            print(f"[bootstrap] !!! CONTEXT OVERFLOW after {self.max_retries} retries — prompt "
                  f"exceeds the vLLM server context. Raise --max-model-len (writer needs the "
                  f"whole evidence batch) or lower --evidence-tok-budget. Detail: {detail[:240]}",
                  flush=True)
        else:
            print(f"[bootstrap] chat completion failed after {self.max_retries} retries: {last_err}",
                  flush=True)
        return None  # infra failure marker: callers must not treat it as model output


# ---------------------------------------------------------------------- #
# Prompt construction (mirrors AlfWorldEnvironmentManager.build_text_obs)
# ---------------------------------------------------------------------- #

def extract_task(text_obs):
    """Replicated from AlfWorldEnvironmentManager.extract_task."""
    task_start = text_obs.find('Your task is to: ')
    if task_start == -1:
        raise ValueError("Task description not found in text observation.")
    return text_obs[task_start + len('Your task is to: '):].strip()


def format_admissible_actions(admissible_actions):
    """Replicated from build_text_obs: exclude 'help', quote each action."""
    return "\n ".join(f"'{s}'" for s in admissible_actions if s != 'help')


def format_action_history(history, history_length):
    """
    Replicated from SimpleMemory.fetch (agent_system/memory/memory.py):
    the last `history_length` (obs, action) pairs, 1-indexed by absolute step.
    """
    recent = history[-history_length:] if history_length > 0 else []
    valid_len = len(recent)
    start_idx = len(history) - valid_len
    lines = []
    for j, rec in enumerate(recent):
        step_num = start_idx + j + 1
        lines.append(
            f"[Observation {step_num}: '{rec['text_obs']}', Action {step_num}: '{rec['action']}']"
        )
    return "\n".join(lines), valid_len


def build_prompt(task, memory_context, current_observation, admissible_actions,
                 history, history_length, inject_style='items', doc_text=''):
    """
    Build the per-step observation prompt.

    inject_style (crossover experiments isolating whether the skill-block
    RENDERING, not its content, flips the with-skill delta):
      items (default): current behavior — retrieved skills formatted with
        markdown section headers into the "## Retrieved Relevant Experience"
        slot of ALFWORLD_TEMPLATE_WITH_MEMORY.
      doc:  OLD faithful-harness rendering (GenericAgent-RL
        skill_slot.patch:88-102): doc_text is a single flowing block glued
        RIGHT AFTER the task sentence of the plain training template, NO
        section headers — "...Your task is to: {task}\\n{doc}\\nPrior to
        this step...". Injected every step, including step 1.
      none: bare arm with the TRAINING-NATIVE prompt — plain
        ALFWORLD_TEMPLATE with no memory section at all (tests whether the
        items-style "No relevant skills found for this task." filler is
        itself a confound).

    DELIBERATE DEVIATION from upstream SkillRL (items and doc styles):
    upstream's step-1 prompt uses ALFWORLD_TEMPLATE_NO_HIS, which omits the
    retrieved skills (they only enter the prompt from step 2 onward via
    ALFWORLD_TEMPLATE_WITH_MEMORY). Here we inject the skills from step 1 as
    well, so every action of the frozen policy is conditioned on the current
    library/doc. This is an intentional difference, marked for the paper's
    protocol section.
    """
    reformatted_admissible_actions = format_admissible_actions(admissible_actions)
    action_history, valid_len = format_action_history(history, history_length)
    step_count = len(history)
    if inject_style == 'items':
        return ALFWORLD_TEMPLATE_WITH_MEMORY.format(
            task_description=task,
            retrieved_memories=memory_context,
            step_count=step_count,
            history_length=valid_len,
            action_history=action_history,
            current_step=step_count + 1,
            current_observation=current_observation,
            admissible_actions=reformatted_admissible_actions,
        )
    # 'doc' / 'none': plain ALFWORLD_TEMPLATE base. For 'doc' the text is
    # spliced into the task line, reproducing skill_slot.patch's
    # "{task_description}{skill_doc}" rendering with skill_doc = "\n" + doc
    # (env_manager._load_skill_doc in the old harness).
    if inject_style == 'doc' and doc_text:
        task_description = f"{task}\n{doc_text}"
    else:
        task_description = task
    return ALFWORLD_TEMPLATE.format(
        task_description=task_description,
        step_count=step_count,
        history_length=valid_len,
        action_history=action_history,
        current_step=step_count + 1,
        current_observation=current_observation,
        admissible_actions=reformatted_admissible_actions,
    )


# ---------------------------------------------------------------------- #
# Rollouts
# ---------------------------------------------------------------------- #

_ACTION_TAG_RE = re.compile(r'\[\s*(/?)\s*action\s*\]', re.IGNORECASE)


def normalize_action_tags(text):
    """Rewrite [action]...[/action] to the canonical <action>...</action>.
    Some models (observed: Qwen2.5-7B-Instruct) emit the square-bracket form
    despite the prompt asking for angle brackets, which alfworld_projection
    cannot parse -> every step invalid -> ~0% success (found 2026-07-09; 3B/1.5B
    use angle brackets and are unaffected). Cheap, idempotent, only touches the
    action delimiter."""
    if not text:
        return text
    text = _ACTION_TAG_RE.sub(
        lambda m: '</action>' if m.group(1) else '<action>', text)
    # Salvage an opened-but-unclosed action tag: Qwen2.5-7B often emits
    # "[action] go to X" (or <action> ...) with NO closing tag at temp>0 / in
    # long-context steps, which leaves no </action> -> invalid. Close it so the
    # command still parses (found 2026-07-09; this was the bulk of the 7B's
    # n_invalid ~43/50 after the bracket + Chinese fixes).
    low = text.lower()
    if '<action>' in low and '</action>' not in low:
        text = text.rstrip() + '</action>'
    return text


def play_game(env, chat, memory, temperature, args):
    """
    Play one ALFWorld game to completion. Returns a trajectory record.

    Protocol matches AlfWorldEnvironmentManager: prompts built from the repo
    templates, actions extracted with alfworld_projection (<action> tags,
    invalid -> truncated raw text which the env answers with
    "Nothing happens."), success = info['won'], reward = 10 * won
    (envs.py compute_reward).
    """
    with ENV_LOAD_LOCK:
        obs, infos = env.reset()
    text_obs = obs[0]
    task = extract_task(text_obs)
    task_type = detect_task_type(task)
    gamefile = infos.get('extra.gamefile', [None])[0]
    admissible = infos['admissible_commands'][0]

    # Retrieve skills once per episode, like the env manager does at reset().
    # Only the 'items' style renders the library; 'doc' and 'none' bypass the
    # retrieval machinery for prompt-building entirely.
    memory_context = ""
    if args.inject_style == 'items':
        retrieved = memory.retrieve(task_description=task, top_k=args.top_k)
        memory_context = memory.format_for_prompt(retrieved)

    history = []      # [{'text_obs': ..., 'action': ...}] for prompt building
    trajectory = []   # [{'action': ..., 'observation': ...}] for the skill writer
    won = False
    n_invalid = 0
    n_inadmissible = 0
    n_api_errors = 0

    # Per-episode doc override (--doc-map-file): E carriers are episode-indexed
    # by construction, so the doc is looked up by gamefile. A miss injects the
    # EMPTY doc (doc branch then renders the bare template) and is auditable
    # via doc_chars in the record.
    episode_doc = getattr(args, 'doc_text', '')
    doc_map = getattr(args, 'doc_map', None)
    if doc_map is not None:
        episode_doc = doc_map.get(doc_map_key(gamefile), '')

    for _ in range(args.max_steps):
        prompt = build_prompt(task, memory_context, text_obs, admissible,
                              history, args.history_length,
                              inject_style=args.inject_style,
                              doc_text=episode_doc)
        completion = chat(prompt, temperature=temperature,
                          max_tokens=args.max_new_tokens)
        if completion is None:
            # API/infra failure, not a model decision. Play on with an empty
            # completion (env answers "Nothing happens.") but mark the
            # episode: it is excluded from skill-writer evidence, since infra
            # noise is not a skill lesson (evolve_common.render_rollouts).
            n_api_errors += 1
            completion = ""
        completion = normalize_action_tags(completion)
        actions, valids = alfworld_projection([completion], [admissible])
        action = actions[0]
        if not valids[0]:
            n_invalid += 1
        elif action.strip().lower() not in {a.strip().lower() for a in admissible}:
            # Well-formed tags, but this command does not exist in the current state
            # ("go to cabinet 4" with three cabinets). The env answers "Nothing happens."
            # and the step is burned exactly like an invalid one -- yet alfworld_projection
            # never notices: its `action_pools` argument is a DEAD PARAMETER, read nowhere
            # in the body. That is fine in verl-agent, where an illegal action is punished
            # through the return rather than a format penalty, but this harness inherited
            # it into frozen-policy EVALUATION, where nothing else punishes it. So
            # n_invalid_actions has been a LOWER BOUND on wasted steps all along.
            # Diagnostic only: `valids` and `action` are untouched, behaviour is unchanged.
            n_inadmissible += 1

        # step() also parses (textgen grammar derive) via the same global
        # tatsu parser -> must hold the lock too. Cheap (ms) next to the
        # concurrent LLM calls, which dominate wall time.
        with ENV_LOAD_LOCK:
            obs, scores, dones, step_infos = env.step([action])
        # Store the pre-step observation with the projected action, matching
        # env_manager's memory.store({'text_obs': pre_text_obs, 'action': actions}).
        history.append({'text_obs': text_obs, 'action': action})
        trajectory.append({'action': action, 'observation': obs[0]})

        text_obs = obs[0]
        admissible = step_infos['admissible_commands'][0]
        won = bool(step_infos['won'][0])
        if dones[0]:
            break

    return {
        'task': task,
        'task_type': task_type,
        'gamefile': gamefile,
        'won': won,
        'reward': 10.0 * float(won),
        'n_steps': len(trajectory),
        'n_invalid_actions': n_invalid,
        'n_inadmissible_actions': n_inadmissible,
        'n_api_errors': n_api_errors,
        'doc_chars': len(episode_doc),
        'trajectory': trajectory,
    }


# ---------------------------------------------------------------------- #
# Opt-in PROCESS-based parallelism (--env-parallel process).
#
# The thread path serializes every construct/reset/step behind ENV_LOAD_LOCK
# because textworld's PDDL grammar parser is a module-global tatsu instance
# shared by all threads in one process; the expensive resets (~2.7s) therefore
# run STRICTLY SERIALLY, bottlenecking each rollout/gate round.
#
# In process mode each worker is a separate PROCESS with its OWN textworld/
# tatsu module (separate address space) — the parser is no longer shared, so
# resets run truly in parallel. ENV_LOAD_LOCK is still acquired inside
# play_game, but each single-threaded child never contends for it (free).
#
# Only PICKLABLE data crosses the process boundary: the config dict, the split
# name, a list of game-file path strings, ints/floats, the args Namespace (all
# primitives), and the skill library as a plain dict. The live env / OpenAI
# client / memory objects are NEVER pickled — each worker reconstructs its own.
# ---------------------------------------------------------------------- #

def _build_child_env(config, split, game_files, seed):
    """Construct a textworld env over `game_files` inside a worker process.

    Bypasses AlfredTWEnv.__init__/collect_game_files (the game list is already
    known, so the disk scan is skipped) by allocating with __new__ and setting
    exactly the attributes init_env reads (config, train_eval, use_expert,
    game_files). Mirrors run_listed_rollouts' copy.copy(base_env) + game_files
    override, but without pickling the live base_env.
    """
    env_cls = get_environment('AlfredTWEnv')
    base = env_cls.__new__(env_cls)
    base.config = config
    base.train_eval = split
    base.use_expert = False
    base.game_files = list(game_files)
    base.num_games = len(game_files)
    env = base.init_env(batch_size=1)
    env.seed(seed)
    return env


def _memory_from_skills(skills):
    """Rebuild a template-mode SkillsOnlyMemory from a plain skills dict.

    The child receives the library as a picklable dict (not a path), so we
    replicate SkillsOnlyMemory.__init__'s template-mode defaults directly —
    exactly how main() builds it (SkillsOnlyMemory(path), all defaults).
    play_game only READS memory (retrieve / format_for_prompt) and only under
    inject_style='items'.
    """
    mem = SkillsOnlyMemory.__new__(SkillsOnlyMemory)
    mem.skills = skills
    mem.retrieval_mode = "template"
    mem.embedding_model_path = "Qwen/Qwen3-Embedding-0.6B"
    mem.task_specific_top_k = None
    mem._embedding_model = None
    mem._skill_embeddings_cache = None
    return mem


def _process_rollout_worker(payload):
    """Top-level (PICKLABLE) ProcessPoolExecutor entrypoint: one worker process.

    Reconstructs its OWN env + ChatClient + memory from the picklable payload,
    plays `payload['count']` games (the env's seeded gamefile cycle over
    payload['game_files']), and returns the list of per-episode record dicts —
    the SAME shape the thread worker returns, so callers are path-agnostic.
    """
    args = payload['args']
    env = _build_child_env(payload['config'], payload['split'],
                           payload['game_files'], payload['seed'])
    chat = ChatClient(payload['base_url'], payload['model'])
    memory = _memory_from_skills(payload['library_skills'])
    records = []
    try:
        for _ in range(payload['count']):
            records.append(play_game(env, chat, memory,
                                     payload['temperature'], args))
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
    return records


def _run_process_pool(base_env, memory, worker_specs, temperature, args):
    """Run per-worker (game_files, count, seed) specs in a ProcessPoolExecutor.

    Each spec becomes one child process. Returns the flat list of records in
    worker order (pool.map preserves submission order), identical in shape to
    the thread path's flattened result.
    """
    payloads = [
        {
            'config': base_env.config,
            'split': base_env.train_eval,
            'game_files': list(game_files),
            'count': count,
            'seed': wseed,
            'temperature': temperature,
            'args': args,
            'base_url': args.base_url,
            'model': args.model,
            'library_skills': memory.skills,
        }
        for (game_files, count, wseed) in worker_specs
    ]
    with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
        results = list(pool.map(_process_rollout_worker, payloads))
    return [rec for worker_records in results for rec in worker_records]


def run_rollouts(base_env, chat, memory, n_games, temperature, seed, args):
    """
    Play `n_games` games with a thread pool of `args.workers` workers.

    Each worker owns one AlfredTWEnv instance seeded `seed + worker_idx` and
    plays a FIXED number of games, so a given (seed, n_games, workers) triple
    always visits the same multiset of games. The gate relies on this: it
    calls run_rollouts with the same fixed seed every round and for both
    conditions, so both arms of the paired comparison see identical games.
    """
    workers = max(1, min(args.workers, n_games))
    counts = [n_games // workers + (1 if i < n_games % workers else 0)
              for i in range(workers)]

    if args.env_parallel == 'process':
        # Same worker->seed->count mapping as the thread path (each worker draws
        # counts[i] games from the FULL split's cycle seeded seed+i), just in a
        # separate process → parallel resets, no ENV_LOAD_LOCK contention.
        worker_specs = [(base_env.game_files, counts[i], seed + i)
                        for i in range(workers)]
        return _run_process_pool(base_env, memory, worker_specs, temperature, args)

    def worker_fn(worker_idx):
        with ENV_LOAD_LOCK:
            env = base_env.init_env(batch_size=1)
            env.seed(seed + worker_idx)
        records = []
        for _ in range(counts[worker_idx]):
            records.append(play_game(env, chat, memory, temperature, args))
        env.close()
        return records

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(worker_fn, range(workers)))
    return [rec for worker_records in results for rec in worker_records]


def run_listed_rollouts(base_env, chat, memory, game_files, temperature, seed,
                        args, label):
    """
    Play an EXPLICIT list of game files exactly once each.

    The list is sharded round-robin across workers. Each worker gets a
    shallow copy of the base env restricted to its shard (init_env registers
    self.game_files at call time, so overriding the attribute on the copy is
    sufficient) and resets exactly len(shard) times: textworld's seeded
    gamefile iterator (TextworldBatchGymEnv.seed -> shuffled_cycle) yields
    each game of the list exactly once per cycle, so order within a shard is
    shuffled but coverage is exact. Used by --enumerate (whole split) and by
    the stratified gate (fixed type-balanced window).
    """
    n_games = len(game_files)
    workers = max(1, min(args.workers, n_games))
    shards = [game_files[i::workers] for i in range(workers)]  # round-robin

    if args.env_parallel == 'process':
        # Same shard->seed mapping as the thread path (worker i plays shards[i]
        # seeded seed+i, each game exactly once), just in a separate process →
        # parallel resets, no ENV_LOAD_LOCK contention. The coverage assertion
        # below runs on the returned records regardless of path.
        worker_specs = [(shards[i], len(shards[i]), seed + i)
                        for i in range(workers)]
        records = _run_process_pool(base_env, memory, worker_specs,
                                    temperature, args)
    else:
        def worker_fn(worker_idx):
            shard = shards[worker_idx]
            with ENV_LOAD_LOCK:
                shard_env = copy.copy(base_env)
                shard_env.game_files = shard
                shard_env.num_games = len(shard)
                env = shard_env.init_env(batch_size=1)
                env.seed(seed + worker_idx)
            records = []
            for _ in range(len(shard)):
                records.append(play_game(env, chat, memory, temperature, args))
            env.close()
            return records

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(worker_fn, range(workers)))
        records = [rec for worker_records in results for rec in worker_records]

    # Coverage assertion: every listed game file exactly once.
    expected = Counter(game_files)
    played = Counter(r['gamefile'] for r in records)
    if played == expected:
        print(f"[bootstrap] COVERAGE_OK: {n_games} games ({label}) "
              f"played exactly once each", flush=True)
    else:
        missing = expected - played
        duplicates = played - expected
        print(f"[bootstrap] WARNING: coverage FAILED ({label}): "
              f"{sum(missing.values())} missing, "
              f"{sum(duplicates.values())} duplicated (of {n_games} expected; "
              f"{len(records)} records) — results are NOT an exact play-once set!",
              file=sys.stderr, flush=True)
        if missing:
            print(f"[bootstrap]   missing: {sorted(missing)[:5]} ...",
                  file=sys.stderr, flush=True)
        if duplicates:
            print(f"[bootstrap]   duplicated: {sorted(duplicates)[:5]} ...",
                  file=sys.stderr, flush=True)
    return records


def run_enumerated_rollouts(base_env, chat, memory, temperature, seed, args):
    """
    Play EVERY game file of the base env's split exactly once (paper-protocol
    exact enumeration, e.g. the 140 valid_seen games). The split's game_files
    list (collected by AlfredTWEnv.collect_game_files) is sorted for
    determinism, then delegated to run_listed_rollouts.
    """
    game_files = sorted(base_env.game_files)
    return run_listed_rollouts(base_env, chat, memory, game_files, temperature,
                               seed, args, label=f"split '{base_env.train_eval}'")


def summarize_records(records):
    """Overall and per-task-type success rates."""
    per_type = {}
    for rec in records:
        stats = per_type.setdefault(rec['task_type'], {'n': 0, 'won': 0})
        stats['n'] += 1
        stats['won'] += int(rec['won'])
    per_type_rates = {
        t: {'success_rate': s['won'] / s['n'], 'n': s['n'], 'won': s['won']}
        for t, s in per_type.items()
    }
    n_total = len(records)
    overall = sum(int(r['won']) for r in records) / n_total if n_total else 0.0
    return overall, per_type_rates


# ---------------------------------------------------------------------- #
# Skill proposal (adapted from SkillUpdater / skill_generation/alfworld.py)
# ---------------------------------------------------------------------- #

def next_dyn_index(current_skills):
    """Copied from SkillUpdater._next_dyn_index."""
    max_idx = 0
    pattern = re.compile(r'^dyn_(\d+)$')
    for skill in current_skills.get('general_skills', []):
        m = pattern.match(skill.get('skill_id', ''))
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    for skills in current_skills.get('task_specific_skills', {}).values():
        for skill in skills:
            m = pattern.match(skill.get('skill_id', ''))
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def reassign_dyn_ids(skills, start_idx):
    """Copied from SkillUpdater._reassign_dyn_ids."""
    reassigned = []
    for i, skill in enumerate(skills):
        updated = dict(skill)
        updated['skill_id'] = f"dyn_{start_idx + i:03d}"
        reassigned.append(updated)
    return reassigned


def all_skill_ids(bank):
    """Copied from SkillsOnlyMemory._get_all_skill_ids (dict version)."""
    ids = set()
    for s in bank.get('general_skills', []):
        if s.get('skill_id'):
            ids.add(s['skill_id'])
    for task_skills in bank.get('task_specific_skills', {}).values():
        for s in task_skills:
            if s.get('skill_id'):
                ids.add(s['skill_id'])
    return ids


def task_specific_skill_ids(bank):
    """
    IDs the writer may revise/delete. GENERAL SECTION FROZEN: self-evolution
    only ever writes task-specific skills — adds go to the weak type's
    category, and revise/delete may only target task-specific IDs — so
    general_skills is never touched by this harness. (o3-library evals still
    read their general section normally through SkillsOnlyMemory.retrieve().)
    """
    ids = set()
    for task_skills in bank.get('task_specific_skills', {}).values():
        for s in task_skills:
            if s.get('skill_id'):
                ids.add(s['skill_id'])
    return ids


def _extract_json_array(response):
    """Return the first JSON array in `response`, or None.

    Replaces a first-'[' .. last-']' slice (2026-07-09). The writer routinely appends
    prose AFTER the array, and that prose contains its own bracket:

        [ {"skill_id": "dyn_003", ...} ]
        This skill helps the agent [verify the object] before acting.

    The old slice ran to the LAST ']', so json.loads saw the array followed by prose and
    raised "Extra data: line 4 column 1 (char 254)" -- a perfectly good skill thrown away
    by the parser. Measured on itbo3_1p5b (15543065): 15 parse failures, 0 gate rejections,
    0 dedup drops -- i.e. the 1.5B library was capped by THIS, not by the model or the gate.

    raw_decode consumes exactly one JSON value and ignores whatever follows, so trailing
    prose is harmless. We try every '[' because the first one may sit inside <think> prose.

    Prefer the first array that actually holds objects: reasoning prose can contain a
    well-formed but irrelevant array (e.g. "check cabinets [1, 2, 3]"), and returning that
    would report parsed_ok with zero skills -- silently downgrading a real proposal to
    NO_CHANGE. Only if no array holds a dict do we fall back to the first array found, which
    keeps the writer's literal `[]` NO_CHANGE answer working.
    """
    dec = json.JSONDecoder()
    first_list = None
    for i, ch in enumerate(response or ""):
        if ch != '[':
            continue
        try:
            value, _ = dec.raw_decode(response[i:])
        except ValueError:
            continue
        if not isinstance(value, list):
            continue
        if any(isinstance(e, dict) for e in value):
            return value
        if first_list is None:
            first_list = value
    return first_list


def parse_skills_response(response, existing_ids=None):
    """
    Extract the JSON array (as SkillUpdater._parse_skills_response does) and
    split its entries into (adds, revisions, deletions).

    An entry without an "op" field (or with op == "add") is a NEW skill and
    must carry skill_id/title/principle, as before. With --allow-revisions the
    writer may also emit {"op": "revise", "skill_id", title/principle/
    when_to_apply} or {"op": "delete", "skill_id"}. Revise/delete ops whose
    skill_id does not exist in the current library are dropped with a log
    line.

    Returns (adds, revisions, deletions, parsed_ok). parsed_ok=True with all
    three lists empty is the writer's first-class NO_CHANGE answer (an empty
    JSON array []): a valid "no proposal this round", NOT a parse failure.
    """
    existing_ids = existing_ids or set()
    adds, revisions, deletions = [], [], []
    parsed_ok = False
    try:
        entries = _extract_json_array(response)
        if entries is not None:
            parsed_ok = True
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                op = entry.get('op', 'add')
                if op == 'add':
                    if all(k in entry for k in ['skill_id', 'title', 'principle']):
                        adds.append({k: v for k, v in entry.items() if k != 'op'})
                elif op in ('revise', 'delete'):
                    skill_id = entry.get('skill_id')
                    if skill_id not in existing_ids:
                        print(f"[bootstrap] dropping invalid {op} op: skill_id "
                              f"{skill_id!r} not in current library", flush=True)
                        continue
                    if op == 'revise':
                        fields = {k: entry[k] for k in ('title', 'principle', 'when_to_apply')
                                  if k in entry}
                        if not fields:
                            print(f"[bootstrap] dropping revise op for {skill_id}: "
                                  f"no title/principle/when_to_apply fields", flush=True)
                            continue
                        revisions.append({'skill_id': skill_id, **fields})
                    else:
                        deletions.append({'skill_id': skill_id})
                else:
                    print(f"[bootstrap] dropping entry with unknown op {op!r}", flush=True)
    except json.JSONDecodeError as e:
        print(f"[bootstrap] JSON parse error in skill response: {e}", flush=True)
    return adds, revisions, deletions, parsed_ok


def format_trajectory(steps):
    """Copied from SkillUpdater._format_trajectory (obs truncated to 200 chars)."""
    lines = []
    for step in steps:
        action = step.get('action', 'unknown')
        obs = step.get('observation', '')[:200]
        lines.append(f"  Action: {action}\n  Observation: {obs}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------- #
# Mechanical grounding filter — independent reproduction of Boer's Meta-era
# verb filter (the original implementation is lost); calibration via the
# `audit` strictness telemetry replaces his manual tuning. Instead of
# prompt-level self-censorship ("only write syntax you saw"), let the writer
# be bold and mechanically screen skills whose action verbs lack support.
# Division of labor: bold writer / mechanical truth filter / paired gate.
# This is what killed the fabricated "toggle the microwave"-style skills in
# the original harness — while `lenient` spares exploration hints, the deep
# over-strictness tension Boer hit at Meta.
# ---------------------------------------------------------------------- #

# Canonical ALFWorld action verbs (env-legal), from the admissible-command
# surface forms ("go to X", "take X from Y", "put X in/on Y", "open/close X",
# "toggle X", "heat/cool/clean X with Y", "use X", "examine X", "look",
# "inventory", "slice X with Y") — cf. envs.py ALF_ACTION_LIST.
ALFWORLD_ACTION_VERBS = (
    'go', 'take', 'put', 'open', 'close', 'toggle', 'heat', 'cool',
    'clean', 'examine', 'use', 'look', 'inventory', 'slice',
)

# Curated NON-env action verbs: words that read as agent procedures but are
# OUTSIDE ALFWorld's action vocabulary — the signature of a fabricated
# procedure (real-world common sense leaking in). Kept deliberately small
# and unambiguous; paraphrases of legal actions (grab/place/move) are NOT
# listed, since dropping those would be false positives under `lenient`.
FABRICATED_ACTION_VERBS = (
    # 'search' removed 2026-07-08 after offline calibration on evolved
    # libraries + the o3 bank: it is strategy vocabulary ("search every
    # surface once"), not a fabricated env command — it dropped 24% of the
    # GOOD o3 bank vs 11% of the harmful loose library (inverted
    # separation, pure false positives).
    'wait', 'wash', 'rinse', 'press', 'push', 'pull', 'unlock', 'plug',
    'pour', 'fill', 'boil', 'preheat', 'activate', 'insert',
)

DETECTABLE_ACTION_VERBS = ALFWORLD_ACTION_VERBS + FABRICATED_ACTION_VERBS


def extract_action_verbs(text, lexicon=DETECTABLE_ACTION_VERBS):
    """
    Lexicon action verbs mentioned in a text. Conservative on purpose:
    word-boundary match on the curated lexicon only (so 'use' does not fire
    on 'house', and non-verb prose words are never flagged).
    """
    low = (text or '').lower()
    return {v for v in lexicon if re.search(r'\b' + v + r'\b', low)}


def evidence_action_verbs(evidence_records):
    """Verbs actually present in this round's evidence trajectories' actions."""
    verbs = set()
    for rec in evidence_records:
        for step in rec.get('trajectory', []):
            verbs |= extract_action_verbs(step.get('action', ''))
    return verbs


def grounding_filter(skills, revisions, evidence_verbs, strictness='lenient'):
    """
    Screen proposed/revised skills by their mentioned action verbs.

    strict:  every mentioned verb must appear in THIS round's evidence
             trajectory actions (original Meta spec). Kills invented
             procedures AND exploration hints.
    lenient: a verb passes if it is in the evidence OR is env-legal
             (ALFWORLD_ACTION_VERBS); only verbs outside the env's action
             vocabulary (e.g. "wait", "search") drop the skill. Blocks true
             fabrications while sparing exploration hints.
    audit:   drops nothing; flags every skill that WOULD be dropped, with
             would_drop_under: ['strict'] / ['strict', 'lenient'] — used to
             measure false-positive rates before choosing a default.

    Returns (kept_skills, kept_revisions, filtered) — filtered entries keep
    the full skill text plus the unsupported verb lists, for the round log.
    """
    env_legal = set(ALFWORLD_ACTION_VERBS)

    kept_skills, kept_revisions, filtered = [], [], []
    for kind, entries, kept in (('add', skills, kept_skills),
                                ('revise', revisions, kept_revisions)):
        for entry in entries:
            text = ' '.join(str(entry.get(k, '')) for k in
                            ('title', 'principle', 'when_to_apply'))
            mentioned = extract_action_verbs(text)
            bad_strict = sorted(mentioned - evidence_verbs)
            bad_lenient = sorted(mentioned - evidence_verbs - env_legal)
            drop = ((strictness == 'strict' and bad_strict) or
                    (strictness == 'lenient' and bad_lenient))
            if drop:
                bad = bad_strict if strictness == 'strict' else bad_lenient
                print(f"[bootstrap] grounding filter dropped "
                      f"{entry.get('skill_id')}: verb '{bad[0]}' not in "
                      f"evidence (strictness={strictness}, unsupported: {bad})",
                      flush=True)
                filtered.append({'op': kind, 'dropped': True,
                                 'strictness': strictness,
                                 'unsupported_strict': bad_strict,
                                 'unsupported_lenient': bad_lenient, **entry})
            else:
                kept.append(entry)
                if strictness == 'audit' and bad_strict:
                    would = ['strict'] + (['lenient'] if bad_lenient else [])
                    filtered.append({'op': kind, 'dropped': False,
                                     'would_drop_under': would,
                                     'unsupported_strict': bad_strict,
                                     'unsupported_lenient': bad_lenient,
                                     **entry})
    return kept_skills, kept_revisions, filtered


def build_skill_prompt(task_type, failed_records, success_records,
                       current_skills, max_new_skills, start_dyn_idx,
                       allow_revisions=True, rejection_note=None,
                       writer_style='direct', grounding='filter',
                       writer_grounded=False):
    """
    Skill-writing prompt for the frozen policy.

    The wording is adapted as closely as possible from the repo's own
    generators: SkillUpdater._build_analysis_prompt (failure analysis, dyn_
    ID handling, JSON-array output contract) and
    skill_generation/alfworld.py::generate_task_specific_skills (task
    description header, success/failure contrast, skill quality criteria).
    This alignment with SkillRL's o3 prompts is scientifically deliberate.

    With allow_revisions, the writer may additionally propose gated
    revise/delete operations on this task type's existing skills — the
    item-granularity correction path that upstream's append-only design
    lacks. All ops go through the same gate as new skills.
    """
    # Evidence rendering keeps the TAIL of each trajectory ([-10:]): failure
    # loops live at the END (evolve_common._fmt_actions keeps head+tail for
    # the same reason); obs are truncated to 200 chars in format_trajectory.
    failure_examples = []
    for i, rec in enumerate(failed_records[:5]):
        failure_examples.append(
            f"\nExample {i + 1}:\n"
            f"Task: {rec['task']}\n"
            f"Task Type: {rec['task_type']}\n"
            f"Trajectory (last 10 steps):\n"
            f"{format_trajectory(rec['trajectory'][-10:])}\n"
        )

    success_examples = []
    for i, rec in enumerate(success_records[:3]):
        success_examples.append(
            f"\nExample {i + 1}:\n"
            f"Task: {rec['task']}\n"
            f"Task Type: {rec['task_type']}\n"
            f"Trajectory (last 10 steps):\n"
            f"{format_trajectory(rec['trajectory'][-10:])}\n"
        )
    success_block = ''.join(success_examples) if success_examples else "\n(none available)\n"

    existing_titles = [s['title'] for s in current_skills.get('general_skills', [])]
    for tt, skills in current_skills.get('task_specific_skills', {}).items():
        for s in skills:
            existing_titles.append(f"[{tt}] {s.get('title', '')}")

    example_ids = ", ".join(
        f'"dyn_{start_dyn_idx + j:03d}"' for j in range(max_new_skills)
    )

    # Prompt-level grounding rules only in 'prompt'/'both' modes. In
    # 'filter'/'off' the strict self-censorship lines are dropped (they made
    # the writer nearly mute: acceptall arm 12 rounds -> 2 skills) — the
    # mechanical grounding_filter is the truth layer instead.
    prompt_grounding = grounding in ('prompt', 'both')

    # Optional revise/delete instructions over this task type's existing
    # skills (id + title + principle shown so the writer can target them).
    revision_block = ""
    has_existing = bool(current_skills.get('task_specific_skills', {}).get(task_type))
    if allow_revisions and has_existing:
        current_type_skills = [
            {
                'skill_id': s.get('skill_id', ''),
                'title': s.get('title', ''),
                'principle': s.get('principle', ''),
            }
            for s in current_skills.get('task_specific_skills', {}).get(task_type, [])
        ]
        max_ops = max_new_skills + 2
        revision_block = f"""
EXISTING {task_type.upper()} SKILLS (you may revise or delete these):
{json.dumps(current_type_skills, indent=2)}

Besides proposing NEW skills, you may also propose operations on the existing skills listed above:
- To revise a skill that is wrong, vague, or unhelpful, include: {{"op": "revise", "skill_id": "<existing id>", "title": "...", "principle": "...", "when_to_apply": "..."}}
- To delete a skill that is harmful or redundant, include: {{"op": "delete", "skill_id": "<existing id>"}}
New-skill entries need no "op" field. Only use skill_ids that appear in the list above for revise/delete.
{'''Only propose a revision or deletion if the trajectories above give concrete evidence that the skill is wrong or harmful; if you see no such evidence, propose no revisions or deletions.
''' if prompt_grounding else ''}Propose at most {max_ops} operations in total (new skills + revisions + deletions).
"""

    # Rejection memory: last paired-gate rejection for this task type (ported
    # from evolve_common.propose:136-137: "a previous edit was REJECTED ...
    # Try a DIFFERENT improvement.").
    rejection_line = f"\n{rejection_note}\n" if rejection_note else ""

    # 2026-07-09. The block below is NAMED "grounding rules" and does the opposite: it forbids
    # the writer from naming anything. Its own example, "to cool an object: pick it up, then use
    # the fridge", is not even a command. Every library we evolved is downstream of it --
    #     1.5B/3B/7B itemized: "Verify Object Location First", "Verify Object Presence", ...
    #     7B whole-doc:        cool <obj> with <cooling_location> ; go to <obj>_location
    # -- while the hand-written gold doc simply names `fridge 1` / `microwave 1` / `sinkbasin 1`
    # and beats the bare 7B by +18.6pp (p=8.7e-7). So the "teacher gap" may be an artifact of
    # this instruction, not a capability gap. --writer-grounded flips exactly this one rule and
    # nothing else; it is OFF by default so existing libraries keep their provenance.
    # threaded as a parameter: referencing the module global `args` here raised
    # NameError whenever the writer path ran from a caller that received args
    # as a parameter (Meta-side find, 2026-07-12)
    if writer_grounded:
        _grounding_rule = (
            "GROUND every command. Copy an identifier LITERALLY when every trajectory above uses "
            "the same one for that thing. When the trajectories show the same kind of thing under "
            "different numbers, keep it generic instead -- the agent picks the right one from the "
            "admissible-action list it is given each turn. NEVER replace an identifier you "
            "actually saw with an invented placeholder such as <cooling_location> or "
            "<obj>_location: the agent cannot execute a placeholder.")
    else:
        _grounding_rule = (
            "NEVER include a specific game's exact object names; GENERALIZE to a CLASS of tasks "
            "(e.g. \"to cool an object: pick it up, then use the fridge\").")

    # Style rules always apply; the "syntax must appear in trajectories"
    # self-censorship line only in prompt-grounding modes.
    if prompt_grounding:
        grounding_block = f"""Grounding rules (violations make skills harmful, not just useless):
- EVERY command syntax you write MUST appear in the trajectories shown above. Do NOT invent procedures from real-world common sense (e.g. do not add open/close/wait steps for an appliance unless the trajectories actually contain them).
- {_grounding_rule}
- Prefer METHOD-LEVEL, generalizable principles over one-off quirks or raw action dumps."""
        no_change_line = ("If the trajectories above show no clear, reusable lesson, "
                          "returning an empty JSON array [] is an acceptable answer "
                          "— do not force a proposal.")
    else:
        grounding_block = f"""Style rules:
- {_grounding_rule}
- Prefer METHOD-LEVEL, generalizable principles over one-off quirks or raw action dumps."""
        no_change_line = ("If the evidence truly shows no reusable lesson, an empty "
                          "array [] is acceptable — but when in doubt, propose your "
                          "best candidate; a validation gate will test it.")

    return f"""Analyze these failed agent trajectories and suggest NEW skills to add to the skill bank.

Task Type: {task_type.upper()}
Description: {TASK_DESCRIPTIONS.get(task_type, '')}

FAILED TRAJECTORIES:
{''.join(failure_examples)}

SUCCESSFUL TRAJECTORIES:
{success_block}

EXISTING SKILL TITLES (avoid duplicating these):
{existing_titles}

Generate 1-{max_new_skills} NEW actionable skills that would help avoid these failures. These should be:
1. **Concise** - 1-2 sentences max per skill
2. **Specific** - Apply specifically to {task_type} tasks
3. **Actionable** - Clear steps or decision rules
4. **Pattern-based** - Identify what makes success vs failure

{grounding_block}

Each skill must have: skill_id, title (3-5 words), principle (1-2 sentences), when_to_apply.

Use skill_ids: {example_ids}
{revision_block}{rejection_line}
{"First, briefly analyze (in 2-4 sentences) what the failed trajectories have in common and what the successful ones did differently. Then output the JSON array of skills." if writer_style == 'reason' else "Return ONLY a JSON array of skills, no other text."}
{no_change_line}
Example format:
[{{"skill_id": "dyn_{start_dyn_idx:03d}", "title": "Verify Object Location First", "principle": "Before attempting to pick up an object, always verify its current location by examining the environment.", "when_to_apply": "When the task requires moving an object but its location is uncertain"}}]
"""


def propose_skills(chat, task_type, failed_records, success_records,
                   current_skills, max_new_skills, start_dyn_idx, args,
                   rejection_note=None):
    """
    Ask the SAME frozen model to write candidate skill operations. Up to 2
    retries on JSON parse failure — but an empty JSON array [] is a valid
    first-class NO_CHANGE answer, accepted without retrying. dyn_ IDs of NEW
    skills are reassigned on our side to guarantee uniqueness regardless of
    what the model returned (as SkillUpdater does). Returns
    (adds, revisions, deletions, raw, grounding_filtered).

    With --grounding filter/both, adds and revisions then pass the
    mechanical grounding_filter against THIS call's evidence trajectories
    (failed + successful) at --grounding-strictness.

    Total operations are capped at max_new_skills + 2 (adds themselves at
    max_new_skills), trimming excess revise/delete ops in response order.
    """
    prompt = build_skill_prompt(task_type, failed_records, success_records,
                                current_skills, max_new_skills, start_dyn_idx,
                                allow_revisions=args.allow_revisions,
                                rejection_note=rejection_note,
                                writer_style=args.writer_style,
                                grounding=args.grounding,
                                writer_grounded=getattr(args, 'writer_grounded', False))
    # GENERAL SECTION FROZEN: only task-specific IDs are valid revise/delete
    # targets, so ops aimed at general_skills are dropped at parse time.
    existing_ids = task_specific_skill_ids(current_skills)
    raw_response = ""
    adds, revisions, deletions = [], [], []
    for attempt in range(3):  # 1 attempt + up to 2 retries on parse failure
        # C2 alignment: the itemized writer used to sample at args.rollout_temp
        # (0.7); it now uses --proposer-temp (default 0.9), the SAME temperature
        # as the doc writer, so temperature is not a confound between skill-style
        # arms (this changes the itemized writer temp 0.7 -> 0.9, deliberate).
        raw_response = chat(prompt, temperature=args.proposer_temp,
                            max_tokens=args.skill_max_new_tokens)
        if raw_response is None:
            raw_response = ""  # writer infra failure -> treat as parse failure
        adds, revisions, deletions, parsed_ok = parse_skills_response(
            raw_response, existing_ids)
        if adds or revisions or deletions:
            break
        if parsed_ok:
            # Valid empty array: the writer explicitly proposes NO_CHANGE.
            print(f"[bootstrap] writer proposed NO_CHANGE for {task_type}",
                  flush=True)
            break
        print(f"[bootstrap] skill proposal parse failure for {task_type} "
              f"(attempt {attempt + 1}/3)", flush=True)
        # Off by default. On itbo3_1p5b (15543065) 9 of 15 parse failures emitted no
        # '[' at all, and the raw text was never kept -- so we could not tell whether
        # the 1.5B writes prose instead of skills, or writes skills and forgets the
        # brackets. Those are different findings. SEV_DEBUG_WRITER=1 makes them visible.
        if os.environ.get("SEV_DEBUG_WRITER"):
            print(f"[bootstrap] [DBG_WRITER_RAW] {task_type} attempt "
                  f"{attempt + 1}: {(raw_response or '')[:700]!r}", flush=True)
    if not args.allow_revisions:
        # Ignore any revise/delete ops the model produced anyway.
        revisions, deletions = [], []
    adds = reassign_dyn_ids(adds, start_dyn_idx)

    # Mechanical grounding filter (before the budget caps, so a dropped
    # skill does not waste a budget slot). Deletions carry no procedure
    # text, so they are never filtered.
    grounding_filtered = []
    if args.grounding in ('filter', 'both'):
        evidence_verbs = evidence_action_verbs(
            list(failed_records) + list(success_records))
        adds, revisions, grounding_filtered = grounding_filter(
            adds, revisions, evidence_verbs,
            strictness=args.grounding_strictness)

    adds = adds[:max_new_skills]
    ops_budget = max_new_skills + 2 - len(adds)
    n_ops = len(revisions) + len(deletions)
    if n_ops > ops_budget:
        print(f"[bootstrap] trimming {n_ops - max(0, ops_budget)} excess "
              f"revise/delete ops for {task_type}", flush=True)
        revisions = revisions[:max(0, ops_budget)]
        deletions = deletions[:max(0, ops_budget - len(revisions))]
    return adds, revisions, deletions, raw_response, grounding_filtered


def find_skill_in_bank(bank, skill_id):
    """Return the live skill dict for skill_id, or None."""
    for skill in bank.get('general_skills', []):
        if skill.get('skill_id') == skill_id:
            return skill
    for task_skills in bank.get('task_specific_skills', {}).values():
        for skill in task_skills:
            if skill.get('skill_id') == skill_id:
                return skill
    return None


def revise_skill_in_bank(bank, revision):
    """Apply one revise op in place; returns True if the skill was found."""
    skill_id = revision['skill_id']
    fields = {k: v for k, v in revision.items() if k != 'skill_id'}
    for skill in bank.get('general_skills', []):
        if skill.get('skill_id') == skill_id:
            skill.update(fields)
            return True
    for task_skills in bank.get('task_specific_skills', {}).values():
        for skill in task_skills:
            if skill.get('skill_id') == skill_id:
                skill.update(fields)
                return True
    return False


def delete_skill_from_bank(bank, skill_id):
    """Dict version of SkillsOnlyMemory.remove_skill; returns True if removed."""
    removed = False
    original_len = len(bank.get('general_skills', []))
    bank['general_skills'] = [
        s for s in bank.get('general_skills', [])
        if s.get('skill_id') != skill_id
    ]
    removed |= len(bank['general_skills']) < original_len
    for task_type in bank.get('task_specific_skills', {}):
        original_len = len(bank['task_specific_skills'][task_type])
        bank['task_specific_skills'][task_type] = [
            s for s in bank['task_specific_skills'][task_type]
            if s.get('skill_id') != skill_id
        ]
        removed |= len(bank['task_specific_skills'][task_type]) < original_len
    return removed


def assert_general_frozen(bank, candidates):
    """
    GENERAL SECTION FROZEN invariant: no op in the candidate batch may touch
    general_skills. Adds always go to a task-specific category, and
    revise/delete IDs were already restricted to task-specific skills at
    parse time — this assertion is the apply-path backstop.
    """
    general_ids = {s.get('skill_id') for s in bank.get('general_skills', [])}
    for cand in candidates:
        assert cand['task_type'] in TASK_TYPES, \
            f"add target must be a task-specific category, got {cand['task_type']!r}"
        for op in list(cand.get('revisions', [])) + list(cand.get('deletions', [])):
            assert op['skill_id'] not in general_ids, \
                f"op targets general skill {op['skill_id']!r} — general section is frozen"


def apply_candidate_ops(bank, candidates):
    """
    Apply the full candidate batch (revisions, then deletions, then adds) to
    a skill bank dict in place. Used both to build the gate's candidate
    library and — on acceptance — kept semantically identical to what
    happens to the live library. Never touches general_skills.
    """
    assert_general_frozen(bank, candidates)
    for cand in candidates:
        for revision in cand.get('revisions', []):
            revise_skill_in_bank(bank, revision)
        for deletion in cand.get('deletions', []):
            delete_skill_from_bank(bank, deletion['skill_id'])
        bank.setdefault('task_specific_skills', {}) \
            .setdefault(cand['task_type'], []) \
            .extend(cand['skills'])


def _all_library_skills(skills_dict):
    """Flatten general + all task-specific skills into one list (for dedup)."""
    out = list(skills_dict.get('general_skills', []) or [])
    for v in (skills_dict.get('task_specific_skills', {}) or {}).values():
        out.extend(v or [])
    return out


def is_duplicate_skill(chat, new_skill, existing_skills):
    """LLM-judge dedup: ask the served policy model whether new_skill restates the
    core advice of an existing skill (a paraphrase, even if worded differently).
    One cheap temp-0 chat. Returns True if the model says duplicate. Used instead
    of embedding cosine because Qwen3-Embedding is not cached and
    sentence-transformers is not installed in the run venv (found 2026-07-08)."""
    if not existing_skills:
        return False
    existing_lines = "\n".join(
        f"{i + 1}. {s.get('title', '')}: {s.get('principle', '')}"
        for i, s in enumerate(existing_skills))
    new_line = f"{new_skill.get('title', '')}: {new_skill.get('principle', '')}"
    prompt = (
        "You are deduplicating a skill library. Here are the EXISTING skills:\n"
        f"{existing_lines}\n\n"
        f"NEW candidate skill:\n{new_line}\n\n"
        "Does the NEW candidate give essentially the SAME core advice as any existing "
        "skill above (a paraphrase or restatement, even if worded differently)? "
        "Answer with ONLY one word: YES (it duplicates an existing skill) or NO "
        "(it adds genuinely new advice)."
    )
    resp = chat(prompt, temperature=0.0, max_tokens=8)
    return bool(resp) and resp.strip().upper().startswith("YES")


def _propose_gate_bestof(base_env, chat, memory, task_type, failed, succeeded,
                         dyn_idx, args, rejection_note, out_dir, round_idx,
                         round_grounding_filtered):
    """itemized best-of-N (--items-n-cands, per_type scope): propose N candidate
    skills for one type (temp>0 gives variety), GATE EACH vs the same base on the
    concentrated window, return the variant with the highest net wins that clears
    the bar (or None). The chosen variant then re-enters the normal single-candidate
    gate/accept path unchanged, so nothing downstream needs to know about best-of-N."""
    per_call_new = min(1, args.max_new_skills)
    variants = []
    for _ in range(max(1, args.items_n_cands)):
        skills, revisions, deletions, raw, filtered = propose_skills(
            chat, task_type, failed, succeeded, memory.skills,
            per_call_new, dyn_idx, args, rejection_note=rejection_note)
        round_grounding_filtered.extend(
            {'task_type': task_type, **f} for f in filtered)
        if args.dedup_judge and skills:
            existing = _all_library_skills(memory.skills)
            kept = []
            for s in skills:
                if is_duplicate_skill(chat, s, existing):
                    print(f"[bootstrap] dedup-judge dropped [{task_type}] "
                          f"'{s.get('title')}' (restates an existing skill)", flush=True)
                else:
                    kept.append(s)
                    existing = existing + [s]
            skills = kept
        if skills or revisions or deletions:
            variants.append({'task_type': task_type, 'skills': skills,
                             'revisions': revisions, 'deletions': deletions,
                             'raw_response': raw})
    if not variants:
        return None
    if len(variants) == 1:
        return variants[0]
    best, best_net = None, None
    shared_base = None      # --gate-shared-base: play the WITHOUT arm once, reuse for all N
    audit = []
    for v in variants:
        sc = run_paired_gate(base_env, chat, memory, [v], args, out_dir, round_idx,
                             records_without=shared_base)
        if args.gate_shared_base and shared_base is None:
            shared_base = sc['records_without']
        t = (sc.get('per_type') or {}).get(task_type, {})
        n_t = t.get('games', 0)
        net = t.get('wins_with', 0) - t.get('wins_without', 0)
        req = max(args.gate_min_net_wins, math.ceil(args.gate_min_delta * n_t))
        title = v['skills'][0].get('title') if v['skills'] else '(rev/del only)'
        passed = n_t > 0 and net >= req
        print(f"[bootstrap] best-of-{args.items_n_cands} [{task_type}] '{title}': "
              f"net {net:+d}/{n_t} (need {req:+d}) {'PASS' if passed else 'fail'}",
              flush=True)
        audit.append({'title': title, 'task_type': task_type, 'net': net,
                      'games': n_t, 'required': req, 'passed': passed,
                      'wins_without': t.get('wins_without', 0),
                      'wins_with': t.get('wins_with', 0),
                      'shared_base': bool(args.gate_shared_base)})
        if passed and (best_net is None or net > best_net):
            best, best_net = v, net
            # scalars only: `sc` also carries records_without (48 full episodes), which
            # must never reach evolve_log.jsonl. The caller pops this key before use.
            best['_gate_score'] = {k: val for k, val in sc.items()
                                   if k != 'records_without'}

    # Persist every candidate's raw tallies. Without this, quantifying the winner's curse
    # (3B: 7 winners advanced, only 2 survived the re-gate) or re-deriving the accept
    # decision under a different bar means re-running the whole 12-round evolution.
    with open(os.path.join(out_dir, f'bestof_audit_round{round_idx}.jsonl'), 'a') as af:
        for row in audit:
            af.write(json.dumps(row) + '\n')
    return best


# ---------------------------------------------------------------------- #
# Whole-doc skill management (--skill-style doc). OLD GenericAgent-RL
# pipeline, ported verbatim-in-spirit from evolve_common.py: a mixed-batch
# whole-doc writer (ALFWORLD_OPTIMIZE_PROMPT) replaces the itemized per-type
# JSON writer, extract_doc replaces parse_skills_response, and run_doc_gate
# (do-no-harm on the whole doc) replaces the itemized paired gate. The doc is
# injected through the existing inject_style='doc' path, fed the EVOLVED doc.
# ---------------------------------------------------------------------- #

# The original ALFWorld optimizer prompt, ported verbatim from
# evolve_common.py:12-27 (run_evolve_alfworld.py:42-57). Used as the SYSTEM
# message of the whole-doc writer call.
ALFWORLD_OPTIMIZE_PROMPT = """You maintain a SHORT skill document: reusable tips/procedures for an ALFWorld agent solving household tasks in a text world (it reads an observation + a list of admissible actions each turn and picks one action).

This document is injected VERBATIM into the agent's prompt, so output the skill content only.

You will see the CURRENT document and several recent attempts (some SOLVED, some FAILED), each with its task and the action sequence taken. Propose ONE small, bounded improvement.

Rules:
- SMALL bounded edit: add or refine AT MOST 1-2 tips. Keep the WHOLE document concise (well under ~1500 tokens). You may edit/merge/delete lines to stay short.
- GENERALIZE to a CLASS of tasks: describe reusable procedures (e.g. "to cool an object: pick it up, then `cool <obj> with fridge 1`"). NEVER include a specific game's exact object names or full action sequence.
- LEARN FROM BOTH: SOLVED attempts -> the procedure that worked; FAILED attempts -> what to avoid / how to fix (e.g. it placed an object before transforming it, or looped picking-and-dropping).
- PREFER RECURRING lessons over one-off quirks.
- METHOD-LEVEL, not a raw action dump.
- If the CURRENT document is EMPTY, you MUST add at least one genuinely reusable tip from a SOLVED attempt -- do not output NO_CHANGE.
- Output ONLY NO_CHANGE if the document already covers the reusable patterns here.
- EVERY command syntax you write MUST appear in the attempts shown. Do NOT invent procedures from real-world common sense (e.g. do not add open/close/wait steps for an appliance unless the attempts actually contain them).
- First reason inside <think> </think> tags: check which exact commands the SOLVED attempts used and where the FAILED ones got stuck. ALL analysis stays inside <think>. After </think>, output the FULL updated document and nothing else -- no headers like "Updated Document:", no commentary -- or exactly NO_CHANGE."""

_FENCE = re.compile(r"^```[a-zA-Z]*\n(.*)\n```$", re.DOTALL)
_DOC_MARKER = re.compile(r"\*{0,3}\s*updated\s+(?:skill\s+)?document\s*:?\s*\*{0,3}", re.IGNORECASE)
_META_START = re.compile(r"^(based on|looking at|from the|analyzing|i will|i'll|the current document|"
                         r"the agent|the most consistent|here is|here's|"
                         # reasoning-trace leaks: weak models narrate instead of emitting a doc.
                         # Observed leak (doc_1p5b round7): "think: In the SOLVED attempt... No change."
                         # slipped through because it neither starts with NO_CHANGE nor any prefix above.
                         r"think\s*:|let me\b|it seems\b|in the (solved|failed))", re.IGNORECASE)
# NO_CHANGE detector: the writer's decline can arrive as "NO_CHANGE", "No change.",
# "no changes needed", etc. The old exact-token check (startswith "NO_CHANGE") missed
# every informal variant, so a bare "No change." became a 10-char "doc". Normalize
# spaces/underscores and allow an optional plural.
_NO_CHANGE = re.compile(r"^no[\s_]*changes?\b", re.IGNORECASE)
# Oversize is a SOFT target, not a cliff: a capable model may run a bit long with
# genuinely good content, so accept the whole doc up to 1.5x max_doc_chars (never
# mid-sentence truncate -- a silent cut is worse). Only egregious overflow (>1.5x =
# rambling/repetition, e.g. the 4400-4746c 1.5B rambles at a 1500 target) is dropped.
DOC_OVERSIZE_TOLERANCE = 1.5

# Doc-writer context budgeting (no tokenizer wired -> char proxy). Used by
# write_doc to size the evidence so the whole prompt fits the server context.
DOC_CHARS_PER_TOKEN = 4      # ~4 chars/token proxy for the evidence blob (APPROXIMATE).
DOC_SCAFFOLD_TOKENS = 1200   # reserve for system prompt + current doc + closing instruction.


def classify_doc_reject(raw, max_doc_chars):
    """Diagnostic twin of extract_doc: label WHY a writer response yielded no doc.
    extract_doc itself only returns None, which made the doc arm's silent no-op
    (bug found 2026-07-08) un-diagnosable. Keep the branch ORDER identical to
    extract_doc so the label matches the real reject path."""
    if raw is None:
        return "infra_none"                          # chat() returned None (e.g. 400 overflow)
    text = raw.replace("<|im_end|>", "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    if "<think>" in text:
        return "unclosed_think"                      # ran out of output budget mid-reasoning
    m = _FENCE.match(text)
    if m:
        text = m.group(1).strip()
    parts = _DOC_MARKER.split(text)
    if len(parts) > 1:
        text = parts[-1].strip()
    if not text or _NO_CHANGE.match(text):
        return "no_change"
    if _META_START.match(text):
        return "meta_prose"                          # emitted analysis, not a document
    if len(text) > max_doc_chars * DOC_OVERSIZE_TOLERANCE:
        return f"oversize({len(text)}>{max_doc_chars}x{DOC_OVERSIZE_TOLERANCE})"
    return "ok"                                      # parsed fine -> was a dup or == current doc


def _fmt_actions(actions, max_acts):
    """Head+tail within the same budget: failures usually live at the END (loops), so keep it.
    Ported verbatim from evolve_common._fmt_actions:37-43."""
    if len(actions) <= max_acts:
        return " ; ".join(actions)
    head, tail = max_acts - 10, 10
    return (" ; ".join(actions[:head]) + f" ; ...({len(actions)-head-tail} omitted)... ; "
            + " ; ".join(actions[-tail:]))


def render_rollouts_doc(records, max_acts=30):
    """Port of evolve_common.render_rollouts:69-86, adapted to bootstrap's record
    shape (won->solved, [s['action'] for s in trajectory]->actions,
    n_invalid_actions->invalid-format turns, trajectory[-1]['observation']->last
    observation). [SOLVED|FAILED] task + action sequence; FAILED episodes get
    bounded failure context. Episodes that hit an API/infra error (n_api_errors>0)
    are EXCLUDED — infra noise is not a skill lesson."""
    out = []
    for e in records:
        if e.get('n_api_errors', 0) > 0:
            continue
        solved = bool(e['won'])
        actions = [s['action'] for s in e['trajectory']]
        tag = "SOLVED" if solved else "FAILED"
        s = f"[{tag}] task: {e['task']}\n  actions: {_fmt_actions(actions, max_acts)}"
        if not solved:
            if e.get('n_invalid_actions'):
                s += f"\n  invalid-format turns: {e['n_invalid_actions']}"
            if e['trajectory']:
                last_obs = e['trajectory'][-1].get('observation') or ''
                if last_obs:
                    s += f"\n  last observation: {last_obs[:120]}"
        out.append(s)
    return "\n\n".join(out)


def extract_doc(text, max_doc_chars):
    """Port of evolve_common.extract_doc:94-116 (MAX_DOC_CHARS parameterized as
    --doc-max-chars). Returns the whole-doc string, or None (NO_CHANGE / analysis
    prose / unclosed <think> / oversize)."""
    text = text.replace("<|im_end|>", "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Pre-closed <think>: some models emit reasoning then a bare </think> with NO
    # opening tag, so the paired-strip above misses it. Cut through the last </think>.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    if "<think>" in text:
        return None                              # opened but never closed -> no usable doc
    m = _FENCE.match(text)
    if m:
        text = m.group(1).strip()
    # reasoning that escaped <think>: keep only what follows an explicit doc marker
    parts = _DOC_MARKER.split(text)
    if len(parts) > 1:
        text = parts[-1].strip()
    if not text or _NO_CHANGE.match(text):
        return None
    if _META_START.match(text):
        return None                              # still analysis prose, not a doc
    if len(text) > max_doc_chars * DOC_OVERSIZE_TOLERANCE:
        return None                              # egregiously over -> rambling; drop (mid-sentence cut is worse)
    return text


# --writer-grounded for the whole-doc writer. Same single-rule flip as the itemized writer
# above, and as GenericAgent_RL_logits b3d7352. The default rule made Qwen2.5-7B evolve
#     cool <obj> with <cooling_location> ; go to <obj>_location
# on the old harness (15568832) -- correct control flow, unexecutable commands.
ALFWORLD_OPTIMIZE_PROMPT_GROUNDED = ALFWORLD_OPTIMIZE_PROMPT.replace(
    "NEVER include a specific game's exact object names or full action sequence.",
    "Do NOT copy a specific game's target-object names or a full action sequence.\n"
    "- GROUND every command you write. Copy an identifier LITERALLY when every attempt you were "
    "shown uses the same one for that thing. When the attempts show the same kind of thing under "
    "different numbers, keep it generic instead -- the agent picks the right one from the "
    "admissible-action list it is given each turn. NEVER replace an identifier you actually saw "
    "with an invented placeholder such as `<cooling_location>` or `<obj>_location`: the agent "
    "cannot execute a placeholder.")
assert ALFWORLD_OPTIMIZE_PROMPT_GROUNDED != ALFWORLD_OPTIMIZE_PROMPT, \
    "grounded whole-doc writer patch did not apply"


def write_doc(chat, current_doc, evidence_records, rejection, args, seed=None):
    """Whole-doc writer: mixed-batch evidence -> ALFWORLD_OPTIMIZE_PROMPT (SYSTEM)
    + a user message (current doc + rendered evidence + OLD-style rejection note +
    closing instruction). Ported from evolve_common.propose:124-145. Samples at
    --proposer-temp, max_tokens=--skill-max-new-tokens. Returns (doc, raw): doc is
    the whole-doc string or None (NO_CHANGE / unparseable / infra), raw is the model's
    verbatim response (or None) kept for diagnostics. No tokenizer is wired, so the
    evidence budget uses a ~4-char/token proxy, sized to fit --max-model-len (see
    body) and capped by --evidence-tok-budget (APPROXIMATE)."""
    evidence = render_rollouts_doc(evidence_records, max_acts=30)
    if not evidence:
        return None, None      # all evidence episodes died on infra errors -> no lesson, no proposal
    # Context-aware evidence budget. BUG 2026-07-08: a fixed 12000-tok budget +
    # 1400 output = 8193 overflowed an 8192-token server, so EVERY writer call
    # 400'd and the doc arm silently no-op'd (12 rounds, both 1.5B and 3B). The
    # original evolve_common relied on a hard-coded 16384 context; here we instead
    # size evidence to whatever --max-model-len allows AFTER reserving the writer's
    # output + system/doc/closing scaffolding, then also cap by --evidence-tok-budget.
    # No tokenizer is wired, so this is a ~4-char/token proxy (APPROXIMATE). Net
    # effect: the writer degrades gracefully (less evidence) at any context size
    # instead of failing silently.
    ctx_evidence_tok = args.max_model_len - args.skill_max_new_tokens - DOC_SCAFFOLD_TOKENS
    if ctx_evidence_tok <= 0:
        print(f"[bootstrap] !!! DOC WRITER: --max-model-len {args.max_model_len} is too small for "
              f"{args.skill_max_new_tokens} output tokens + scaffolding; cannot build a writer "
              f"prompt. Raise --max-model-len.", flush=True)
        return None, None
    budget_tok = min(args.evidence_tok_budget, ctx_evidence_tok)
    char_budget = budget_tok * DOC_CHARS_PER_TOKEN
    if len(evidence) > char_budget:
        evidence = evidence[:char_budget] + "\n…(evidence truncated to fit context)…"
    user = (f"CURRENT skill document:\n{current_doc or '(empty)'}\n\n"
            f"Recent attempts:\n{evidence}\n")
    if rejection:
        user += (f"\nNote: a previous edit was REJECTED (held-out score dropped: {rejection}). "
                 "Try a DIFFERENT improvement.\n")
    user += ("\nNow reason inside <think> </think> first, then output the document content only "
             "(no preamble/explanation), or exactly NO_CHANGE.")
    _system = (ALFWORLD_OPTIMIZE_PROMPT_GROUNDED if getattr(args, 'writer_grounded', False)
               else ALFWORLD_OPTIMIZE_PROMPT)
    raw = chat(user, temperature=args.proposer_temp,
               max_tokens=args.skill_max_new_tokens, system=_system,
               seed=seed)
    if raw is None:
        return None, None      # writer infra failure -> no proposal, never crash the run
    return extract_doc(raw, args.doc_max_chars), raw


# ---------------------------------------------------------------------- #
# Gate
# ---------------------------------------------------------------------- #

def run_paired_gate(base_env, chat, memory, candidates, args, out_dir, round_idx,
                    records_without=None):
    """
    Play the SAME fixed, type-STRATIFIED window of gate games twice (fixed
    seed, temp 0 greedy): once with the current library, once with the
    current library + candidate op batch.

    The window is a largest-remainder type-balanced selection over the
    split's game files (stratified_games), fixed across rounds and arms.
    Scores are computed both ways: raw mean, and the proportional-stratified
    mean (per-type mean weighted by native type proportions) so the dominant
    pick_and_place type cannot mask another type's regression. Which one
    drives acceptance is --gate-metric. Returns a dict of both arms' scores.

    CONCENTRATED WINDOW (per_type scope only): a full-split window spreads
    --gate-games across all 6 types (~16/type at 96), but only the candidate
    types' games can differ between arms — the rest are prompt-identical and
    contribute nothing, while vs140 noise is ±5pp single-run. So the window
    pool is restricted to the gamefile prefixes of THIS round's candidate
    types (LIB_TYPE_TO_PREFIXES; largest-remainder among those, floor 1),
    giving each judged type 32-96 games instead of ~16. Same fixed gate seed
    + deterministic selection => the same candidate set yields the same
    window across arms and rounds; the per-round arm-multiset pairing guard
    below still verifies it. The net-wins bar max(1, ceil(gate_min_delta *
    type_games)) automatically tightens with more games. Batch scope keeps
    the full-split window.
    """
    gate_seed = args.seed + GATE_SEED_OFFSET + (round_idx * 100003 if args.gate_fresh_window else 0)

    # Fixed stratified gate window (same seed every round => same games for
    # a given pool and for both arms; run_listed_rollouts plays exactly once
    # each and asserts coverage, which doubles as the pairing guard).
    all_games = sorted(base_env.game_files)
    window_pool = all_games
    window_prefixes = None
    if args.gate_scope == 'per_type':
        candidate_types = {c['task_type'] for c in candidates}
        window_prefixes = sorted({p for t in candidate_types
                                  for p in LIB_TYPE_TO_PREFIXES.get(t, ())})
        pool = [g for g in all_games if game_type(g) in window_prefixes]
        if pool:
            window_pool = pool
        else:
            print(f"[bootstrap] WARNING round {round_idx}: no games for "
                  f"candidate types {sorted(candidate_types)} — falling back "
                  f"to the full-split gate window", file=sys.stderr, flush=True)
            window_prefixes = None
    gate_window, props = stratified_games(window_pool, gate_seed, args.gate_games)
    if window_prefixes is not None:
        print(f"[bootstrap] round {round_idx} gate window concentrated on "
              f"{window_prefixes}: {len(gate_window)} games", flush=True)

    # The WITHOUT arm depends only on (gate_window, memory) -- never on the candidate.
    # best-of-N gates N candidates against the same window and the same library, so it
    # used to replay this identical arm N times. That is not merely wasted compute: vLLM
    # greedy is not bit-reproducible across batches (identical config, two jobs: 16.4% vs
    # 19.3%), so each replay produced a DIFFERENT wins_without, and `net = with - without`
    # inherited that noise. best-of-N then picked argmax(net), i.e. partly the candidate
    # whose base arm happened to score low. Passing records_without in lets the caller
    # compute it once and share it, which both halves the cost and makes the N nets
    # commensurable. Callers that omit it keep the old behaviour exactly.
    if records_without is None:
        records_without = run_listed_rollouts(
            base_env, chat, memory, gate_window, 0.0, gate_seed, args,
            label=f"gate round {round_idx} without-candidates")

    # Candidate library = deep copy of the current bank with the ENTIRE op
    # batch (adds + revisions + deletions) applied, loaded through
    # SkillsOnlyMemory so retrieval/formatting is identical.
    candidate_bank = copy.deepcopy(memory.skills)
    apply_candidate_ops(candidate_bank, candidates)
    candidate_path = os.path.join(out_dir, f"gate_candidate_library_round{round_idx}.json")
    with open(candidate_path, 'w') as f:
        json.dump(candidate_bank, f, indent=2)
    candidate_memory = SkillsOnlyMemory(candidate_path)

    records_with = run_listed_rollouts(
        base_env, chat, candidate_memory, gate_window, 0.0, gate_seed, args,
        label=f"gate round {round_idx} with-candidates")

    # Paired-comparison validity check: both arms must have visited the same
    # game multiset (explicit window makes this structural; keep the guard).
    games_without = sorted(r['gamefile'] or '' for r in records_without)
    games_with = sorted(r['gamefile'] or '' for r in records_with)
    if games_without != games_with:
        print(f"[bootstrap] WARNING round {round_idx}: gate arms saw DIFFERENT "
              f"games ({sum(a != b for a, b in zip(games_without, games_with))} "
              f"mismatches of {len(games_without)}) — paired delta is invalid!",
              file=sys.stderr, flush=True)

    raw_without, _ = summarize_records(records_without)
    raw_with, _ = summarize_records(records_with)

    # Per-LIBRARY-type win tallies for --gate-scope per_type. Grouping by the
    # record's detected task_type (not the gamefile prefix) is what makes the
    # per-type deltas independently attributable: template retrieval routes a
    # type-t skill into exactly the episodes whose goal detects as type t, so
    # non-t games are prompt-identical across the two arms.
    per_type_tallies = {}
    for rec in records_without:
        tally = per_type_tallies.setdefault(
            rec['task_type'], {'games': 0, 'wins_without': 0, 'wins_with': 0})
        tally['games'] += 1
        tally['wins_without'] += int(rec['won'])
    for rec in records_with:
        tally = per_type_tallies.setdefault(
            rec['task_type'], {'games': 0, 'wins_without': 0, 'wins_with': 0})
        tally['wins_with'] += int(rec['won'])

    return {
        'records_without': records_without,   # so best-of-N can reuse the shared base arm
        'n_gate_games': len(gate_window),
        'window_prefixes': window_prefixes,
        'raw_without': raw_without,
        'raw_with': raw_with,
        'stratified_without': proportional_stratified_mean(records_without, props),
        'stratified_with': proportional_stratified_mean(records_with, props),
        'per_type': per_type_tallies,
    }


# ---------------------------------------------------------------------- #
# Whole-doc do-no-harm gate (--skill-style doc)
# ---------------------------------------------------------------------- #

def run_doc_gate(base_env, chat, memory, current_doc, cand_docs, args, round_idx):
    """Do-no-harm gate on the whole doc (port of self_evolve_alfworld.py's
    per-step gate). Plays the SAME fresh (if --gate-fresh-window) UNSTRATIFIED
    window, natural type distribution, matching the old harness (temp 0 greedy)
    once per doc: a WITHOUT arm under current_doc, then one WITH arm per candidate
    doc. Paired — identical window + gate seed across arms, only args.doc_text is
    swapped; the arms run SEQUENTIALLY so the shared args.doc_text read by the
    rollout workers is race-free. args.doc_text is restored afterwards.
    Acceptance (in the caller) is do-no-harm: cand_success >= base_success.
    """
    gate_seed = args.seed + GATE_SEED_OFFSET + (round_idx * 100003 if args.gate_fresh_window else 0)
    # Unstratified random window (natural type distribution), matching the old
    # harness's whole-doc do-no-harm gate. The seed is computed ONCE here so the
    # WITHOUT arm and every WITH arm share the SAME window (paired); with
    # --gate-fresh-window it is round-dependent, re-sampling a fresh window each
    # round to keep the doc from overfitting one fixed game set.
    all_games = sorted(base_env.game_files)
    gate_window = random.Random(gate_seed).sample(
        all_games, min(args.gate_games, len(all_games)))
    saved_doc = getattr(args, 'doc_text', '')
    try:
        args.doc_text = current_doc
        records_base = run_listed_rollouts(
            base_env, chat, memory, gate_window, 0.0, gate_seed, args,
            label=f"doc gate round {round_idx} without (current doc)")
        base_success, _ = summarize_records(records_base)
        cand_successes = []
        for j, cand_doc in enumerate(cand_docs):
            args.doc_text = cand_doc
            records_cand = run_listed_rollouts(
                base_env, chat, memory, gate_window, 0.0, gate_seed, args,
                label=f"doc gate round {round_idx} with (cand {j})")
            cand_successes.append(summarize_records(records_cand)[0])
    finally:
        args.doc_text = saved_doc
    # Acceptance is do-no-harm on PLAIN success, and the window is unstratified,
    # so the stratified fields just mirror plain success (no
    # proportional_stratified_mean).
    base_stratified = base_success
    cand_stratified = list(cand_successes)
    return {
        'n_gate_games': len(gate_window),
        'base_success': base_success,
        'base_stratified': base_stratified,
        'cand_successes': cand_successes,
        'cand_stratified': cand_stratified,
    }


def run_doc_round(base_env, chat, memory, args, out_dir, round_idx,
                  current_doc, doc_rejection, log_path, library_size_before, t_start):
    """One round of the OLD whole-doc pipeline (--skill-style doc): src rollouts
    under the CURRENT doc -> mixed-batch evidence -> whole-doc writer -> do-no-harm
    gate -> doc persisted (evolved_doc key + doc_round{n}.md/doc_latest.md mirrors).
    Returns the (possibly updated) (current_doc, doc_rejection)."""
    # 1. src rollouts with the CURRENT evolved doc injected (inject_style='doc',
    #    byte-identical actor path to the itemized arm; only args.doc_text differs).
    args.doc_text = current_doc
    rollout_seed = args.seed + round_idx * 10000
    if args.enumerate:
        records = run_enumerated_rollouts(base_env, chat, memory,
                                          args.rollout_temp, rollout_seed, args)
    else:
        records = run_rollouts(base_env, chat, memory, args.rollout_games,
                               args.rollout_temp, rollout_seed, args)
    overall_success, per_type = summarize_records(records)
    print(f"[bootstrap] round {round_idx} rollouts: overall success "
          f"{overall_success:.3f} over {len(records)} games | per-type: "
          f"{ {t: round(v['success_rate'], 3) for t, v in per_type.items()} }",
          flush=True)
    with open(os.path.join(out_dir, f'games_round{round_idx}.jsonl'), 'w') as gf:
        for rec in records:
            gf.write(json.dumps({'gamefile': rec['gamefile'],
                                 'task_type': rec['task_type'],
                                 'won': rec['won'],
                                 'n_steps': rec['n_steps'],
                                 'n_invalid': rec['n_invalid_actions'],
                                 'n_inadmissible': rec.get('n_inadmissible_actions', 0),
                                 'doc_chars': rec.get('doc_chars', 0),
                                 'actions': [s['action'] for s in rec['trajectory']]}) + '\n')

    # 2. Mixed-batch evidence: ALL non-API-error records, FLAT (cross-type, no
    #    per-type filter, no [:5]/[:3]) — the OLD whole-doc writer sees the whole
    #    batch (evolve_common.render_rollouts).
    evidence_records = [r for r in records if r['n_api_errors'] == 0]
    n_excluded = len(records) - len(evidence_records)
    if n_excluded:
        print(f"[bootstrap] excluding {n_excluded} API-error episode(s) from "
              f"doc-writer evidence", flush=True)

    # 3. Whole-doc candidates (deduped). Each candidate gets a DISTINCT, reproducible
    #    seed (OLD harness pattern SEED0 + 10000*step + j*17) so n_cands>1 is diverse
    #    AND replayable: same (args.seed, round_idx) -> candidate j always gets the same
    #    seed -> the same doc under vLLM seeded sampling; different j -> different seed.
    cands = []
    writer_raw = []           # per-candidate raw + reject reason (diagnostic; mirrors
                              # the itemized path's writer_silence, added 2026-07-08)
    for j in range(args.doc_n_cands):
        cand_seed = args.seed + round_idx * 10000 + j * 17
        c, raw = write_doc(chat, current_doc, evidence_records, doc_rejection, args, seed=cand_seed)
        appended = bool(c) and c != current_doc and c not in cands
        writer_raw.append({
            'cand': j, 'seed': cand_seed,
            'extracted': bool(c),
            'appended': appended,
            'reject': None if c else classify_doc_reject(raw, args.doc_max_chars),
            'raw': (raw or '')[:2000],
        })
        if appended:
            cands.append(c)
    # Persist the model's own words when NOTHING survived, so a silent no-op round
    # is diagnosable from disk (no more grepping .out for "no candidate doc").
    if not cands:
        with open(os.path.join(out_dir, f'doc_writer_silence_round{round_idx}.json'), 'w') as wf:
            json.dump(writer_raw, wf, indent=2)

    gate_record = {
        'skill_style': 'doc',
        'mode': args.gate_mode,
        'scope': 'batch',
        'min_delta': args.gate_min_delta,
        'games': args.gate_games,
        'n_cands': len(cands),
        'doc_len_before': len(current_doc),
        'base_success': None,
        'cand_successes': None,
        'best_cand_success': None,
        'delta': None,
        'accepted': False,
        'skipped': False,
    }
    accepted = False
    if not cands:
        gate_record['skipped'] = True
        print(f"[bootstrap] round {round_idx} doc: NO_CHANGE (no candidate doc) "
              f"| reject reasons: {[w['reject'] for w in writer_raw]} "
              f"| {time.time()-t_start:.0f}s", flush=True)
    else:
        # 4/5/6. Do-no-harm gate on the stratified window (paired), best cand by
        #        success, accept iff best_cand - base >= --gate-min-delta (0.0 =>
        #        cand >= base, OLD do-no-harm >=).
        scores = run_doc_gate(base_env, chat, memory, current_doc, cands, args, round_idx)
        base_sr = scores['base_success']
        cand_srs = scores['cand_successes']
        best = max(range(len(cands)), key=lambda j: cand_srs[j])
        best_cand, best_sr = cands[best], cand_srs[best]
        take = (best_sr - base_sr) >= args.gate_min_delta   # do-no-harm (>= at min_delta=0)
        gate_record.update({
            'games_played': scores['n_gate_games'],
            'base_success': base_sr,
            'cand_successes': cand_srs,
            'best_cand_idx': best,
            'best_cand_success': best_sr,
            'base_stratified': scores['base_stratified'],
            'best_cand_stratified': scores['cand_stratified'][best],
            'delta': best_sr - base_sr,
            'accepted': bool(take),
        })
        accepted = bool(take)
        if take:
            current_doc = best_cand
            doc_rejection = ''
        else:
            # OLD-style rejection note fed to the next round's writer.
            doc_rejection = f"{base_sr:.0%}->{best_sr:.0%}"
        print(f"[bootstrap] round {round_idx} doc gate: base={base_sr:.3f} "
              f"best_cand={best_sr:.3f} (best of {len(cands)}) -> "
              f"{'ACCEPT' if take else 'REJECT'} | doclen={len(current_doc)} | "
              f"rej='{doc_rejection}' | {time.time()-t_start:.0f}s", flush=True)

    # 7/8. injection + persistence. The evolved doc lives BOTH as
    #      memory.skills['evolved_doc'] (keeps the library_round{n}.json artifact
    #      contract) AND as doc_round{n}.md / doc_latest.md mirrors (OLD-style
    #      resume/inspection).
    memory.skills['evolved_doc'] = current_doc
    memory.skills.setdefault('metadata', {})['self_evolve'] = {
        'generated_by': 'selfevolve/bootstrap.py',
        'skill_style': 'doc',
        'model': args.model,
        'gate_mode': args.gate_mode,
        'rounds_completed': round_idx,
        'seed': args.seed,
    }
    snapshot_path = os.path.join(out_dir, f'library_round{round_idx}.json')
    memory.save_skills(snapshot_path)
    for name in (f'doc_round{round_idx}.md', 'doc_latest.md'):
        with open(os.path.join(out_dir, name), 'w') as f:
            f.write(current_doc)

    wall_time = time.time() - t_start
    round_record = {
        'round': round_idx,
        'skill_style': 'doc',
        'split': args.split,
        'enumerate': args.enumerate,
        'inject_style': args.inject_style,
        'doc_file': args.doc_file,
        'rollout_games': len(records),
        'rollout_temp': args.rollout_temp,
        'proposer_temp': args.proposer_temp,
        'n_api_error_episodes': sum(1 for r in records if r['n_api_errors'] > 0),
        'overall_success': overall_success,
        'per_task_type': per_type,
        'n_doc_cands': len(cands),
        'candidate_docs': cands,
        'writer_reject_reasons': [w['reject'] for w in writer_raw],
        'doc_accepted': accepted,
        'doc_len': len(current_doc),
        'doc_rejection': doc_rejection,
        'gate': gate_record,
        'library_size_before': library_size_before,
        'library_size_after': memory.get_skill_count(),
        'library_snapshot': snapshot_path,
        'wall_time_sec': wall_time,
    }
    with open(log_path, 'a') as f:
        f.write(json.dumps(round_record) + '\n')
    print(f"[bootstrap] round {round_idx} done in {wall_time:.1f}s | "
          f"doclen={len(current_doc)} | log -> {log_path}", flush=True)
    return current_doc, doc_rejection


# ---------------------------------------------------------------------- #
# Main loop
# ---------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen-weights skill self-evolution harness (with gate) for ALFWorld.")
    parser.add_argument('--base-url', default='http://127.0.0.1:8901/v1',
                        help='OpenAI-compatible endpoint of the frozen policy (vLLM).')
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct',
                        help='Served model name.')
    parser.add_argument('--skill-style', choices=['itemized', 'doc'],
                        default='itemized',
                        help='Skill-management pipeline. itemized (default): the '
                             'current per-type JSON-item writer + paired/per-type '
                             'gate + retrieved-items injection. doc: the OLD '
                             'GenericAgent-RL whole-doc pipeline — a mixed-batch '
                             'whole-doc writer (ALFWORLD_OPTIMIZE_PROMPT), '
                             'do-no-harm gate, and whole-doc injection. doc is an '
                             'UMBRELLA switch: it forces --inject-style doc and '
                             '--gate-scope batch and is incompatible with '
                             '--inject-style items / --gate-scope per_type / '
                             '--allow-revisions (explicitly setting those errors; '
                             'unset defaults are silently overridden).')
    parser.add_argument('--proposer-temp', type=float, default=0.9,
                        help='Skill-WRITER sampling temperature, used by BOTH the '
                             'itemized and doc writers. NOTE (C2 alignment): the '
                             'itemized writer previously sampled at --rollout-temp '
                             '(0.7); it now uses --proposer-temp (default 0.9) so '
                             'temperature is not a confound between skill-style '
                             'arms (itemized writer temp changes 0.7 -> 0.9).')
    parser.add_argument('--doc-n-cands', type=int, default=1,
                        help='--skill-style doc only: whole-doc proposals per round '
                             '(deduped; best passing one accepted). ChatClient has '
                             'no seed, so n_cands>1 diversity is unpinned — 1 '
                             'recommended.')
    parser.add_argument('--doc-max-chars', type=int, default=1500,
                        help='--skill-style doc only: hard CHAR cap in extract_doc '
                             '(oversize proposal -> dropped invalid). OLD '
                             'MAX_DOC_CHARS.')
    parser.add_argument('--evidence-tok-budget', type=int, default=12000,
                        help='--skill-style doc only: cap on rendered mixed-batch '
                             'evidence. No tokenizer is wired, so a ~4-char/token '
                             'proxy (~48000 chars) is used — APPROXIMATE. OLD '
                             'EVIDENCE_TOK_BUDGET. Also capped by --max-model-len.')
    parser.add_argument('--max-model-len', type=int, default=16384,
                        help='The vLLM server context window (MUST match the server\'s '
                             '--max-model-len). The doc writer sizes its evidence budget to '
                             'fit within this after reserving output + scaffolding, so it can '
                             'never silently overflow. Default 16384: the whole-doc writer\'s '
                             'evidence batch needs a large context (a 12000-tok budget + 1400 '
                             'out = 8193 overflows an 8192 server — bug found 2026-07-08). The '
                             'actor rollouts use tiny prompts and are unaffected.')
    parser.add_argument('--rounds', type=int, default=8)
    # Formal sizes matching our GenericAgent_RL_logits selfevolve protocol:
    # src=32 (evidence rollouts the skill writer sees), sel=48 (paired gate).
    parser.add_argument('--rollout-games', type=int, default=32,
                        help='ALFWorld games per round from --split (protocol "src", '
                             'formal=32). Ignored under --enumerate.')
    parser.add_argument('--split',
                        choices=['train', 'eval_in_distribution', 'eval_out_of_distribution'],
                        default='train',
                        help='ALFWorld split (AlfredTWEnv train_eval): train (default, '
                             'current behavior), eval_in_distribution (valid_seen), '
                             'eval_out_of_distribution (valid_unseen).')
    parser.add_argument('--enumerate', action='store_true',
                        help='Play every game file of --split exactly once per round '
                             'instead of seeded random game streams (paper-protocol '
                             'eval, e.g. valid_seen-140). Ignores --rollout-games.')
    parser.add_argument('--rollout-temp', type=float, default=0.7)
    parser.add_argument('--gate-games', type=int, default=48,
                        help='Fixed paired-gate game set size, train split, temp 0 '
                             '(protocol "sel", formal=48).')
    parser.add_argument('--gate-mode', choices=['paired', 'accept_all', 'random_matched'],
                        default='paired')
    parser.add_argument('--gate-min-delta', type=float, default=0.0,
                        help='Candidate batch accepted iff paired success delta >= this.')
    parser.add_argument('--gate-min-net-wins', type=int, default=1,
                        help='per_type scope: ABSOLUTE floor on net wins for a type-t '
                             'skill to be accepted -> required = max(--gate-min-net-wins, '
                             'ceil(--gate-min-delta * type_games)). Default 1 = old '
                             'behavior. With --gate-max-types 1 the window concentrates '
                             '~48 games on one type; set this to 2 for a real-but-small '
                             'bar (+2/48 is above a 1-game noise flip yet clearable by a '
                             'weak model) instead of the ceil(0.1*48)=+5 that starved '
                             'item_cg to an empty library.')
    parser.add_argument('--dedup-judge', action='store_true',
                        help='Semantic dedup: before a proposed skill enters the gate/'
                             'library, ask the SERVED policy model (one cheap temp-0 '
                             'chat) whether it restates the core advice of an existing '
                             'skill; drop it if so. No embedding model needed '
                             '(Qwen3-Embedding is not cached / sentence-transformers not '
                             'installed). Targets the item_3b "verify object location x18" '
                             'redundancy.')
    parser.add_argument('--writer-grounded', action='store_true',
                        help="flip ONE rule in BOTH writers (itemized + whole-doc). Default: "
                             "\"NEVER include a specific game's exact object names; GENERALIZE to "
                             'a CLASS of tasks (e.g. "to cool an object: pick it up, then use the '
                             'fridge")\' -- note the example is not even a command. Grounded: name '
                             'appliances/receptacles LITERALLY (`cool <obj> with fridge 1`) and '
                             'forbid invented placeholders. Every platitude we ever evolved '
                             '("Verify Object Location First") and every unexecutable placeholder '
                             '(`cool <obj> with <cooling_location>`) is downstream of the default '
                             'rule, while the gold doc that beats bare 7B by +18.6pp simply names '
                             '`fridge 1`. OFF by default.')
    parser.add_argument('--gate-shared-base', action='store_true',
                        help='best-of-N: play the gate WITHOUT arm once and reuse it for '
                             'every candidate, instead of replaying it per candidate. '
                             'Halves gate cost AND removes a real bias: greedy vLLM is not '
                             'bit-reproducible, so per-candidate base replays gave each '
                             'candidate a different wins_without, and argmax(net) partly '
                             'selected the candidate whose base happened to score low. '
                             'OFF by default so published numbers keep their old protocol.')
    parser.add_argument('--gate-accept-bestof', action='store_true',
                        help='accept the best-of-N winner on its own gate result instead of '
                             're-gating it in the round batch gate. Saves 2 more passes. '
                             'WARNING: that re-gate is the only replication check; measured '
                             'on itbo3_3b, 5 of 7 winners failed to reproduce. With '
                             '--allow-revisions off, a falsely accepted skill is permanent '
                             'and contaminates every later round. Raw per-candidate tallies '
                             'are written to bestof_audit_round*.jsonl either way.')
    parser.add_argument('--items-n-cands', type=int, default=1,
                        help='itemized best-of-N (per_type scope): for the targeted '
                             'type, PROPOSE this many candidate skills (temp>0 gives '
                             'variety), GATE each vs the same base on the concentrated '
                             'window, and KEEP the one with the highest net wins that '
                             'clears the bar. Default 1 = single proposal (existing '
                             'behavior). Mirrors doc best-of-N (--doc-n-cands); costs '
                             '~N extra gate passes per round.')
    parser.add_argument('--gate-metric', choices=['stratified', 'mean'],
                        default='stratified',
                        help='Gate acceptance score: "stratified" = per-type mean '
                             'weighted by native type proportions (one dominant type '
                             'cannot mask another\'s regression; alfworld_env.py:202), '
                             '"mean" = raw success mean. Both are always logged.')
    parser.add_argument('--gate-fresh-window', action='store_true',
                        help='Re-sample the gate window every round (round-dependent '
                             'seed) instead of a fixed window. Prevents the library '
                             'overfitting to one fixed game set; matches the old harness '
                             'design. Applies to both run_paired_gate and run_doc_gate.')
    parser.add_argument('--grounding', choices=['prompt', 'filter', 'both', 'off'],
                        default='filter',
                        help='Anti-hallucination layer. filter (default): relaxed '
                             'writer prompt + mechanical post-parse verb filter '
                             '(bold writer / mechanical truth filter / paired gate). '
                             'prompt: strict prompt rules only (v4 behavior). '
                             'both: strict prompt AND filter. off: neither '
                             '(bold writer, no truth layer; ablation arm).')
    parser.add_argument('--grounding-strictness', choices=['strict', 'lenient', 'audit'],
                        default='lenient',
                        help='Mechanical filter strictness. strict: every mentioned '
                             'lexicon verb must appear in this round\'s evidence '
                             'actions (kills invented procedures AND exploration '
                             'hints). lenient (default): env-legal verbs always pass; '
                             'only verbs outside the ALFWorld action vocabulary '
                             '(e.g. "wait") drop a skill. audit: drop nothing, flag '
                             'would-drop under both levels — telemetry for choosing '
                             'the default.')
    parser.add_argument('--inject-style', choices=['items', 'doc', 'none'],
                        default='items',
                        help='Skill-block RENDERING (crossover experiments: does the '
                             'rendering, not the content, flip the with-skill delta?). '
                             'items (default): retrieved library items with markdown '
                             'headers in the "## Retrieved Relevant Experience" slot. '
                             'doc: --doc-file text as one flowing block glued right '
                             'after the task sentence of the plain training template '
                             '(old faithful-harness rendering; library retrieval '
                             'bypassed for prompts). none: training-native plain '
                             'template, no memory section at all (is the "No relevant '
                             'skills found" filler itself a confound?).')
    parser.add_argument('--doc-map-file', default=None,
                        help="JSON mapping '<taskdir>/<trial>' -> doc text, injected "
                             "PER EPISODE under --inject-style doc (episode-indexed E "
                             "carriers, e.g. oracle object locations). Games absent "
                             "from the map get an EMPTY doc; audit via doc_chars in "
                             "games_round records.")
    parser.add_argument('--doc-file', default=None,
                        help='Path to the skill doc injected verbatim every step '
                             '(including step 1) when --inject-style doc.')
    parser.add_argument('--writer-style', choices=['direct', 'reason'],
                        default='direct',
                        help='direct (default): "Return ONLY a JSON array". '
                             'reason: writer first briefly analyzes failure/'
                             'success patterns, then outputs the JSON array '
                             '(light-reasoning ablation; parser extracts the '
                             'array from surrounding text either way).')
    parser.add_argument('--evidence-channel', choices=['both', 'fail_only'],
                        default='both',
                        help='Writer evidence: both (default, <=5 failed + <=3 '
                             'successful trajectories) or fail_only (SkillRL '
                             'online-updater behavior; ablation arm).')
    parser.add_argument('--gate-scope', choices=['batch', 'per_type'],
                        default='per_type',
                        help='per_type (default): both gate arms run ONCE with all '
                             'candidates, then each type\'s candidate is accepted '
                             'independently by NET WINS among that type\'s gate games '
                             '(>= max(1, ceil(gate-min-delta * type games))) — valid '
                             'because a type-t skill only enters type-t prompts, so '
                             'non-t games are identical across arms. batch: the whole '
                             'op batch is accepted/rejected together by --gate-metric '
                             'delta (>= --gate-min-delta).')
    parser.add_argument('--gate-max-types', type=int, default=0,
                        help='per_type scope only: cap the number of weakest task '
                             'types that propose (and thus get gated) per round to '
                             'this many, so the concentrated gate window (v6) actually '
                             'concentrates 96 games onto few types. 0 = unlimited (all '
                             'weak types, the v6 default). Typical v7 value: 2.')
    parser.add_argument('--max-new-skills', type=int, default=3,
                        help='Per-call ceiling on NEW skills asked of the writer. The '
                             'effective per-call number is driven by the round budget: '
                             'min(1, --max-new-skills, remaining --max-ops-per-round).')
    parser.add_argument('--max-ops-per-round', type=int, default=2,
                        help='BATCH-scope only: total op budget per round (adds + '
                             'revisions + deletions across the worst <=2 weak types, '
                             '1 new skill requested per writer call). With '
                             '--gate-scope per_type each weak type may get 1 proposal '
                             'and this cap does not apply (per-type verdicts are '
                             'independently attributable).')
    parser.add_argument('--allow-revisions', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Let the skill writer also propose gated revise/delete '
                             'ops on existing skills of the weak task type '
                             '(--no-allow-revisions for append-only upstream behavior).')
    parser.add_argument('--weak-threshold', type=float, default=0.4,
                        help='Propose skills only for task types below this success '
                             'rate (use 1.01 to always propose).')
    parser.add_argument('--out-dir', default='selfevolve/out')
    parser.add_argument('--library', default=None,
                        help='Path to initial skill library JSON; default starts EMPTY.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=50,
                        help='Max env steps per episode (config_tw.yaml also caps at 50).')
    parser.add_argument('--smoke', action='store_true',
                        help='2 rollout games, 4 gate games, 1 round — pipeline sanity check.')
    # Protocol knobs, defaults matching examples/grpo_trainer/run_alfworld_skills.sh
    # and verl/trainer/config/ppo_trainer.yaml (env.history_length: 2).
    parser.add_argument('--workers', type=int, default=16,
                        help='Concurrent env/rollout workers (threads, or processes '
                             'under --env-parallel process).')
    parser.add_argument('--env-parallel', choices=['thread', 'process'],
                        default='thread',
                        help='How the --workers env workers run. thread (default, '
                             'exact current behavior): a ThreadPoolExecutor with ALL '
                             'textworld construct/reset/step serialized behind '
                             'ENV_LOAD_LOCK (the module-global tatsu PDDL parser is '
                             'not thread-safe), so the ~2.7s resets run STRICTLY '
                             'SERIALLY. process: a ProcessPoolExecutor — each worker '
                             'is a separate process with its OWN tatsu parser, so '
                             'resets run in parallel and the lock is uncontended. '
                             'Each process rebuilds its own env + ChatClient + memory '
                             'from picklable data; the vLLM endpoint serves concurrent '
                             'HTTP from all processes. Real concurrency is capped by '
                             'CPUs available (SLURM --cpus-per-task).')
    parser.add_argument('--history-length', type=int, default=2)
    parser.add_argument('--top-k', type=int, default=6,
                        help='General skills injected per prompt (skills_only_memory.top_k).')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                        help='Per-step generation budget (data.max_response_length).')
    parser.add_argument('--skill-max-new-tokens', type=int, default=1024,
                        help='Generation budget for skill-writing calls.')
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # --skill-style doc UMBRELLA switch. The whole-doc pipeline requires
    # inject_style='doc' and gate_scope='batch' and is incompatible with the
    # itemized-only knobs. An EXPLICITLY-set incompatible value hard-errors;
    # an unset (default) value is silently overridden — so `--skill-style doc`
    # with everything else at defaults just works. This is a pure CLI-
    # consistency check (no runtime deps), so it runs BEFORE the import guard.
    # ------------------------------------------------------------------ #
    if args.skill_style == 'doc':
        def _flag_set(*names):
            return any(a == n or a.startswith(n + '=')
                       for a in sys.argv[1:] for n in names)
        errors = []
        if _flag_set('--inject-style') and args.inject_style != 'doc':
            errors.append(f"--inject-style {args.inject_style} is incompatible with "
                          f"--skill-style doc (doc mode forces inject_style='doc')")
        if _flag_set('--gate-scope') and args.gate_scope != 'batch':
            errors.append(f"--gate-scope {args.gate_scope} is incompatible with "
                          f"--skill-style doc (doc mode forces gate_scope='batch')")
        if _flag_set('--allow-revisions'):
            errors.append("--allow-revisions is incompatible with --skill-style doc "
                          "(there are no library items to revise in doc mode)")
        if errors:
            for e in errors:
                print(f"[bootstrap] ERROR: {e}", file=sys.stderr)
            sys.exit(2)
        if args.inject_style != 'doc':
            print("[bootstrap] skill-style doc: overriding inject_style "
                  f"'{args.inject_style}' -> 'doc'", flush=True)
            args.inject_style = 'doc'
        if args.gate_scope != 'batch':
            print("[bootstrap] skill-style doc: overriding gate_scope "
                  f"'{args.gate_scope}' -> 'batch'", flush=True)
            args.gate_scope = 'batch'
        args.allow_revisions = False

    if _IMPORT_ERROR is not None:
        print(f"[bootstrap] Missing runtime dependency: {_IMPORT_ERROR}\n"
              f"Run this on a machine with the SkillRL stack installed "
              f"(alfworld/textworld, openai, torch).", file=sys.stderr)
        sys.exit(1)

    if args.smoke:
        args.rounds = 1
        args.rollout_games = 2
        args.gate_games = 4
        print("[bootstrap] SMOKE mode: rounds=1, rollout_games=2, gate_games=4", flush=True)

    # Skill doc for --inject-style doc: read verbatim once, injected every
    # step (incl. step 1). Stripped like the old harness's _load_skill_doc
    # so the "\n{doc}\n" splice has no stray blank lines. Under --skill-style
    # doc, --doc-file is the OPTIONAL round-0 SEED (default '' = OLD doc='');
    # under a bare --inject-style doc (eval/crossover arm) it stays REQUIRED.
    args.doc_text = ''
    args.doc_map = None
    if args.inject_style == 'doc':
        if getattr(args, 'doc_map_file', None):
            with open(args.doc_map_file) as f:
                args.doc_map = json.load(f)
            print(f"[bootstrap] inject-style doc: PER-EPISODE map, "
                  f"{len(args.doc_map)} entries from {args.doc_map_file}", flush=True)
        if args.doc_file:
            with open(args.doc_file) as f:
                args.doc_text = f.read().strip()
            print(f"[bootstrap] inject-style doc: {len(args.doc_text)} chars from "
                  f"{args.doc_file}", flush=True)
        elif args.doc_map is not None:
            pass  # per-episode map supplies the text; no global doc needed
        elif args.skill_style == 'doc':
            print("[bootstrap] skill-style doc: no --doc-file seed, starting from "
                  "an EMPTY doc", flush=True)
        else:
            print("[bootstrap] --inject-style doc requires --doc-file",
                  file=sys.stderr)
            sys.exit(1)
    elif args.doc_file:
        print(f"[bootstrap] WARNING: --doc-file given but --inject-style is "
              f"'{args.inject_style}' — doc ignored", flush=True)
    if args.inject_style != 'items':
        print(f"[bootstrap] inject-style '{args.inject_style}': library/retrieval "
              f"is bypassed for prompt rendering (gate deltas will not reflect "
              f"library candidates; intended for eval/crossover arms)", flush=True)

    random.seed(args.seed)
    coin = random.Random(args.seed + 1)  # random_matched gate coin flips

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'evolve_log.jsonl')

    # ------------------------------------------------------------------ #
    # Skill library (SkillRL's exact JSON container, via SkillsOnlyMemory)
    # ------------------------------------------------------------------ #
    if args.library:
        library_path = args.library
    else:
        library_path = os.path.join(out_dir, 'library_init.json')
        with open(library_path, 'w') as f:
            json.dump(empty_library(), f, indent=2)
    memory = SkillsOnlyMemory(library_path)
    # Guarantee the container shape (all six ALFWorld task keys present) so
    # snapshots stay drop-in compatible with run_alfworld_skills*.sh.
    memory.skills.setdefault('general_skills', [])
    for task_type in TASK_TYPES:
        memory.skills.setdefault('task_specific_skills', {}).setdefault(task_type, [])
    memory.skills.setdefault('common_mistakes', [])
    memory.skills.setdefault('metadata', {})

    # ------------------------------------------------------------------ #
    # Environment: AlfredTWEnv directly (no ray env_manager), repo config +
    # $ALFWORLD_DATA convention. --split maps straight onto AlfredTWEnv's
    # train_eval (same values envs.py:93-96 passes through: train /
    # eval_in_distribution=valid_seen / eval_out_of_distribution=valid_unseen).
    # ------------------------------------------------------------------ #
    config = load_config_file(ALF_CONFIG_PATH)
    env_type = config['env']['type']
    assert env_type == 'AlfredTWEnv', f"Expected AlfredTWEnv, got {env_type}"
    base_env = get_environment(env_type)(config, train_eval=args.split)
    if args.enumerate:
        print(f"[bootstrap] --enumerate: playing all {len(base_env.game_files)} "
              f"games of split '{args.split}' exactly once per round "
              f"(--rollout-games={args.rollout_games} ignored)", flush=True)

    chat = ChatClient(args.base_url, args.model)

    memory.save_skills(os.path.join(out_dir, 'library_round0.json'))

    paired_verdict_history = []  # random_matched acceptance rate (batch scope)
    per_type_verdict_history = {}  # random_matched acceptance rate (per_type scope)
    # Rejection memory: per task type, the last paired-gate rejection of a
    # proposal for that type. Injected into the next writer prompt for the
    # type ("try a DIFFERENT kind of improvement"), cleared on acceptance.
    # Ported from evolve_common.propose (rej note) / self_evolve_alfworld.py.
    last_rejection = {}

    # --skill-style doc: the evolving whole document. Round-0 seed = --doc-file
    # (or '' = OLD doc=''); doc_rejection is the OLD-style "without%->with%" note
    # fed to the next round's writer. Mirror the seed to the round-0 artifacts.
    # Eval/resume: a --library saved from a prior doc run carries its whole doc in
    # the 'evolved_doc' key. Load THAT (else fall back to the --doc-file seed).
    # Bug 2026-07-08: this read args.doc_text unconditionally, so curve-eval with
    # --library (no --doc-file) injected an EMPTY doc -> every round silently scored
    # the BARE model (doc_round0.md came out 0 bytes even when the library held a doc).
    current_doc = args.doc_text
    if args.library and args.skill_style == 'doc':
        with open(args.library) as _lf:
            current_doc = json.load(_lf).get('evolved_doc') or args.doc_text
    doc_rejection = ''
    if args.skill_style == 'doc':
        memory.skills['evolved_doc'] = current_doc
        memory.save_skills(os.path.join(out_dir, 'library_round0.json'))
        for name in ('doc_round0.md', 'doc_latest.md'):
            with open(os.path.join(out_dir, name), 'w') as f:
                f.write(current_doc)

    for round_idx in range(1, args.rounds + 1):
        t_start = time.time()
        print(f"\n[bootstrap] ===== Round {round_idx}/{args.rounds} =====", flush=True)
        library_size_before = memory.get_skill_count()

        # --skill-style doc: whole-doc writer + do-no-harm gate + whole-doc
        # injection (mixed-batch evidence). Everything below is the itemized arm.
        if args.skill_style == 'doc':
            current_doc, doc_rejection = run_doc_round(
                base_env, chat, memory, args, out_dir, round_idx,
                current_doc, doc_rejection, log_path, library_size_before, t_start)
            continue

        # 1. Rollouts with the CURRENT library in the prompt.
        rollout_seed = args.seed + round_idx * 10000
        if args.enumerate:
            records = run_enumerated_rollouts(base_env, chat, memory,
                                              args.rollout_temp, rollout_seed, args)
        else:
            records = run_rollouts(base_env, chat, memory, args.rollout_games,
                                   args.rollout_temp, rollout_seed, args)
        overall_success, per_type = summarize_records(records)
        print(f"[bootstrap] round {round_idx} rollouts: overall success "
              f"{overall_success:.3f} over {len(records)} games | per-type: "
              f"{ {t: round(v['success_rate'], 3) for t, v in per_type.items()} }",
              flush=True)
        # Per-game outcomes to disk: flip-rate analysis across repeated runs
        # (engine nondeterminism at temp 0 measured at ~5pp/192 games on
        # 2026-07-08) is impossible without them. `actions` carries the
        # projected action per step: behavioural probes (e.g. did the policy
        # transform before placing?) read the sequence, never the endpoint score.
        with open(os.path.join(out_dir, f'games_round{round_idx}.jsonl'), 'w') as gf:
            for rec in records:
                gf.write(json.dumps({'gamefile': rec['gamefile'],
                                     'task_type': rec['task_type'],
                                     'won': rec['won'],
                                     'n_steps': rec['n_steps'],
                                     'n_invalid': rec['n_invalid_actions'],
                                     'n_inadmissible': rec.get('n_inadmissible_actions', 0),
                                     'doc_chars': rec.get('doc_chars', 0),
                                     'actions': [s['action'] for s in rec['trajectory']]}) + '\n')

        # 2. Propose candidate skill ops for the WORST weak task types (same
        #    frozen model), under a small per-round op budget.
        #
        #    Rationale: upstream SkillRL adds at most 3 skills per update
        #    event TOTAL (one analyze_failures call per trigger,
        #    ray_trainer.py:924-930) — not per task type. Our earlier
        #    per-weak-type x max_new_skills loop (up to 18 ops/round) was an
        #    unintended amplification, and Boer's original GenericAgent
        #    protocol adds 1-2 skills per evolution step precisely so the
        #    paired gate can attribute the measured delta to a small batch.
        #    So: rank weak types by success rate ascending (ties: fewer games
        #    first), take the worst min(2, len(weak)) types, request ONE new
        #    skill per writer call, and stop once the round total
        #    (adds + revisions + deletions) reaches --max-ops-per-round.
        #
        #    PER-TYPE gate scope relaxes the global cap: because a type-t
        #    skill is only injected into type-t episodes, per-type deltas in
        #    a single paired run are independently attributable, so EVERY
        #    weak type may get 1 proposal and each type is gated on its own
        #    verdict (--max-ops-per-round applies to batch scope only).
        weak_types = [t for t, v in per_type.items()
                      if v['n'] > 0 and v['success_rate'] < args.weak_threshold]
        weak_types.sort(key=lambda t: (per_type[t]['success_rate'], per_type[t]['n']))
        if args.gate_scope == 'per_type':
            target_types = (weak_types if args.gate_max_types <= 0
                            else weak_types[:args.gate_max_types])
            ops_left = None  # no global round cap in per_type scope
        else:
            target_types = weak_types[:min(2, len(weak_types))]
            ops_left = args.max_ops_per_round
        candidates = []
        writer_silence = []
        round_grounding_filtered = []
        dyn_idx = next_dyn_index(memory.skills)
        for task_type in target_types:
            if ops_left is not None and ops_left <= 0:
                break
            # Evidence discipline: episodes that hit an API/infra error are
            # NOT skill lessons — exclude them from writer evidence (they
            # still count in the success stats above). Mirrors
            # evolve_common.render_rollouts skipping e["error"] episodes.
            evidence = [r for r in records
                        if r['task_type'] == task_type and r['n_api_errors'] == 0]
            n_excluded = sum(1 for r in records
                             if r['task_type'] == task_type and r['n_api_errors'] > 0)
            if n_excluded:
                print(f"[bootstrap] excluding {n_excluded} {task_type} episode(s) "
                      f"with API errors from writer evidence", flush=True)
            failed = [r for r in evidence if not r['won']][:5]
            succeeded = [r for r in evidence if r['won']][:3]
            # Evidence-channel ablation: SkillRL's online updater is fail-only;
            # our default shows both (the grounding rules need correct-syntax
            # examples, which only successes provide for a weak writer).
            if args.evidence_channel == 'fail_only':
                succeeded = []
            if not failed:
                continue  # nothing to analyze (possible when weak-threshold > 1)
            # Rejection memory note for this type, if its last proposal was
            # rejected by the paired gate.
            rejection_note = None
            if task_type in last_rejection:
                rej = last_rejection[task_type]
                rejection_note = (
                    f"Note: your previous proposal for this task type was REJECTED: "
                    f"with-skill {rej['success_with']:.1%} vs without "
                    f"{rej['success_without']:.1%} on the paired gate. "
                    f"Try a DIFFERENT kind of improvement.")
            # itemized best-of-N: propose N variants for this type, gate each vs the
            # same base, keep the best-passing. Leaves the single-candidate path below
            # untouched (used when --items-n-cands == 1).
            if args.items_n_cands and args.items_n_cands > 1 and args.gate_scope == 'per_type':
                best = _propose_gate_bestof(
                    base_env, chat, memory, task_type, failed, succeeded, dyn_idx,
                    args, rejection_note, out_dir, round_idx, round_grounding_filtered)
                if best is not None:
                    candidates.append(best)
                    dyn_idx += len(best['skills'])
                else:
                    writer_silence.append({
                        'task_type': task_type,
                        'raw_response': '(best-of-%d: no variant cleared the bar)'
                                        % args.items_n_cands})
                continue
            # Effective per-call new-skill count: 1, still ceilinged by
            # --max-new-skills (and by the remaining round budget in batch scope).
            if ops_left is None:
                per_call_new = min(1, args.max_new_skills)
            else:
                per_call_new = min(1, args.max_new_skills, ops_left)
            skills, revisions, deletions, raw_response, filtered = propose_skills(
                chat, task_type, failed, succeeded, memory.skills,
                per_call_new, dyn_idx, args, rejection_note=rejection_note)
            round_grounding_filtered.extend(
                {'task_type': task_type, **f} for f in filtered)
            if args.dedup_judge and skills:
                existing_for_dedup = _all_library_skills(memory.skills)
                deduped = []
                for s in skills:
                    if is_duplicate_skill(chat, s, existing_for_dedup):
                        print(f"[bootstrap] dedup-judge dropped [{task_type}] "
                              f"'{s.get('title')}' (restates an existing skill)",
                              flush=True)
                    else:
                        deduped.append(s)
                        existing_for_dedup = existing_for_dedup + [s]
                skills = deduped
            if ops_left is not None:
                # Batch scope: enforce the round budget across ALL op kinds
                # (revisions and deletions count toward the same cap),
                # trimming adds first-in.
                skills = skills[:ops_left]
                revisions = revisions[:ops_left - len(skills)]
                deletions = deletions[:ops_left - len(skills) - len(revisions)]
            if skills or revisions or deletions:
                candidates.append({
                    'task_type': task_type,
                    'skills': skills,
                    'revisions': revisions,
                    'deletions': deletions,
                    'raw_response': raw_response,
                })
                dyn_idx += len(skills)
                if ops_left is not None:
                    ops_left -= len(skills) + len(revisions) + len(deletions)
            else:
                # NO_CHANGE (or unparseable): keep the raw writer response —
                # distinguishing reasoned restraint from lazy defaults needs
                # the model's own words (diagnostic gap found 2026-07-08).
                writer_silence.append({
                    'task_type': task_type,
                    'raw_response': (raw_response or '')[:2000],
                })
        # Flat op list for the round log (load-bearing for the paper's
        # author-drift analysis: does the writer start deleting/revising its
        # own earlier skills?).
        ops = []
        for cand in candidates:
            ops.extend({'op': 'add', 'task_type': cand['task_type'], **s}
                       for s in cand['skills'])
            ops.extend({'op': 'revise', 'task_type': cand['task_type'], **r}
                       for r in cand['revisions'])
            ops.extend({'op': 'delete', 'task_type': cand['task_type'], **d}
                       for d in cand['deletions'])
        n_adds = sum(len(c['skills']) for c in candidates)
        print(f"[bootstrap] round {round_idx}: weak types {weak_types} "
              f"(targeting {target_types}), "
              f"{n_adds} new skills / "
              f"{sum(len(c['revisions']) for c in candidates)} revisions / "
              f"{sum(len(c['deletions']) for c in candidates)} deletions proposed "
              f"(round budget {args.max_ops_per_round})", flush=True)

        # 3. Gate the candidate batch. Both arms run ONCE with ALL candidates;
        #    with --gate-scope per_type each type's candidate is then judged
        #    on its own gate-game subset (independently attributable: a
        #    type-t skill only enters type-t prompts, so non-t games are
        #    prompt-identical across arms).
        gate_record = {
            'mode': args.gate_mode,
            'metric': args.gate_metric,
            'scope': args.gate_scope,
            'gate_max_types': args.gate_max_types,
            'min_delta': args.gate_min_delta,
            'games': args.gate_games,
            'success_without': None,
            'success_with': None,
            'raw_without': None,
            'raw_with': None,
            'stratified_without': None,
            'stratified_with': None,
            'delta': None,
            'delta_raw': None,
            'delta_stratified': None,
            'paired_verdict': None,
            'accept_prob': None,
            'per_type': None,
            'accepted': False,
            'skipped': False,
        }
        accepted_candidates = []
        if not candidates:
            gate_record['skipped'] = True
        elif args.gate_mode == 'accept_all':
            # SkillRL behavior: accept unconditionally, no evaluation.
            for c in candidates:
                c.pop('_gate_score', None)   # never let it reach evolve_log.jsonl
            accepted_candidates = list(candidates)
            gate_record['accepted'] = True
            if args.gate_scope == 'per_type':
                gate_record['per_type'] = {
                    c['task_type']: {'accepted': True} for c in candidates}
        else:
            # --gate-accept-bestof: the best-of-N winner was already gated against the
            # shared base on this very window, so replaying that comparison only re-rolls
            # decoding noise. Reuse its score. (It IS a replication check -- see the flag's
            # help -- so this is opt-in, not the default.)
            pregated = [c.pop('_gate_score', None) for c in candidates]
            if (args.gate_accept_bestof and len(candidates) == 1
                    and pregated[0] is not None):
                scores = pregated[0]
                print(f"[bootstrap] round {round_idx}: reusing the best-of-"
                      f"{args.items_n_cands} gate result (no re-gate)", flush=True)
            else:
                scores = run_paired_gate(
                    base_env, chat, memory, candidates, args, out_dir, round_idx)
            delta_raw = scores['raw_with'] - scores['raw_without']
            delta_stratified = (scores['stratified_with']
                                - scores['stratified_without'])
            # Overall scores in --gate-metric units (drive acceptance in
            # batch scope; logged in both scopes).
            if args.gate_metric == 'stratified':
                success_without = scores['stratified_without']
                success_with = scores['stratified_with']
                delta = delta_stratified
            else:
                success_without = scores['raw_without']
                success_with = scores['raw_with']
                delta = delta_raw
            paired_verdict = delta >= args.gate_min_delta
            gate_record.update({
                'games_played': scores['n_gate_games'],
                'window_prefixes': scores['window_prefixes'],
                'success_without': success_without,
                'success_with': success_with,
                'raw_without': scores['raw_without'],
                'raw_with': scores['raw_with'],
                'stratified_without': scores['stratified_without'],
                'stratified_with': scores['stratified_with'],
                'delta': delta,
                'delta_raw': delta_raw,
                'delta_stratified': delta_stratified,
                'paired_verdict': paired_verdict,
            })

            if args.gate_scope == 'batch':
                if args.gate_mode == 'paired':
                    accepted = paired_verdict
                else:  # random_matched: rate-matched control (batch scope).
                    if paired_verdict_history:
                        accept_prob = sum(paired_verdict_history) / len(paired_verdict_history)
                    else:
                        accept_prob = 0.5
                    gate_record['accept_prob'] = accept_prob
                    accepted = coin.random() < accept_prob
                paired_verdict_history.append(paired_verdict)
                gate_record['accepted'] = accepted
                accepted_candidates = list(candidates) if accepted else []
                print(f"[bootstrap] round {round_idx} gate ({args.gate_metric}): "
                      f"without={success_without:.3f} with={success_with:.3f} "
                      f"delta={delta:+.3f} (raw delta {delta_raw:+.3f}) -> "
                      f"{'ACCEPT' if accepted else 'REJECT'}", flush=True)
                # Rejection memory: on rejection remember the gate scores per
                # proposed task type; on acceptance clear those types.
                for cand in candidates:
                    if accepted:
                        last_rejection.pop(cand['task_type'], None)
                    else:
                        last_rejection[cand['task_type']] = {
                            'round': round_idx,
                            'success_with': success_with,
                            'success_without': success_without,
                        }
            else:
                # PER-TYPE scope: each type's candidate is judged by NET WINS
                # among that type's gate games. --gate-min-delta was
                # calibrated as a fraction of the mixed gate set (0.021 * 48
                # = 1 net win); per-type subsets are far smaller, so the
                # criterion is expressed in net wins directly:
                # accept iff wins_with - wins_without >=
                #     max(1, ceil(gate_min_delta * type_games)).
                per_type_detail = {}
                for cand in candidates:
                    t = cand['task_type']
                    tally = scores['per_type'].get(
                        t, {'games': 0, 'wins_without': 0, 'wins_with': 0})
                    n_t = tally['games']
                    net_wins = tally['wins_with'] - tally['wins_without']
                    required = max(args.gate_min_net_wins,
                                   math.ceil(args.gate_min_delta * n_t))
                    type_verdict = n_t > 0 and net_wins >= required
                    detail = {
                        'games': n_t,
                        'wins_without': tally['wins_without'],
                        'wins_with': tally['wins_with'],
                        'net_wins': net_wins,
                        'required_net_wins': required,
                        'paired_verdict': type_verdict,
                    }
                    if args.gate_mode == 'paired':
                        type_accepted = type_verdict
                    else:  # random_matched, per-type rate-matched control
                        hist = per_type_verdict_history.get(t, [])
                        accept_prob = sum(hist) / len(hist) if hist else 0.5
                        detail['accept_prob'] = accept_prob
                        type_accepted = coin.random() < accept_prob
                    per_type_verdict_history.setdefault(t, []).append(type_verdict)
                    detail['accepted'] = type_accepted
                    per_type_detail[t] = detail
                    if type_accepted:
                        accepted_candidates.append(cand)
                        last_rejection.pop(t, None)
                    elif n_t > 0:
                        # Revisions/deletions targeting type t ride with t's
                        # verdict; remember the per-type rates for the note.
                        last_rejection[t] = {
                            'round': round_idx,
                            'success_with': tally['wins_with'] / n_t,
                            'success_without': tally['wins_without'] / n_t,
                        }
                    print(f"[bootstrap] round {round_idx} gate[{t}]: "
                          f"{tally['wins_without']}->{tally['wins_with']} wins "
                          f"of {n_t} games (net {net_wins:+d}, need "
                          f"{required:+d}) -> "
                          f"{'ACCEPT' if type_accepted else 'REJECT'}", flush=True)
                gate_record['per_type'] = per_type_detail
                gate_record['accepted'] = bool(accepted_candidates)
                paired_verdict_history.append(paired_verdict)

        # 4. Apply the ACCEPTED candidates' ops to the live library (all of
        #    them in batch scope, the per-type winners in per_type scope).
        #    Rejected -> nothing changes. Adds/deletes go through the
        #    SkillsOnlyMemory methods (dedup, embedding-cache invalidation);
        #    the net effect matches apply_candidate_ops on the gate copy.
        #    GENERAL SECTION FROZEN: same invariant as the gate copy.
        if accepted_candidates:
            assert_general_frozen(memory.skills, accepted_candidates)
            for cand in accepted_candidates:
                pt_verdict = (gate_record.get('per_type') or {}).get(cand['task_type'], {})
                for revision in cand['revisions']:
                    # Provenance: record the pre-revision fields on the skill
                    # itself before overwriting (lineage without log replay).
                    old = find_skill_in_bank(memory.skills, revision['skill_id'])
                    if old is not None:
                        old.setdefault('provenance', {}).setdefault('revisions', []).append({
                            'round': round_idx,
                            'old_title': old.get('title'),
                            'old_principle': old.get('principle'),
                        })
                    if revise_skill_in_bank(memory.skills, revision):
                        memory._skill_embeddings_cache = None
                        print(f"[bootstrap] Revised skill: {revision['skill_id']}",
                              flush=True)
                for deletion in cand['deletions']:
                    memory.remove_skill(deletion['skill_id'])
                for s in cand['skills']:
                    s['provenance'] = {
                        'created_round': round_idx,
                        'gate_mode': args.gate_mode,
                        'gate_net_wins': pt_verdict.get('net_wins'),
                        'writer_model': args.model,
                    }
                memory.add_skills(cand['skills'], category=cand['task_type'])

        # 5. Snapshot + JSONL log.
        memory.skills.setdefault('metadata', {})['self_evolve'] = {
            'generated_by': 'selfevolve/bootstrap.py',
            'model': args.model,
            'gate_mode': args.gate_mode,
            'rounds_completed': round_idx,
            'seed': args.seed,
        }
        snapshot_path = os.path.join(out_dir, f'library_round{round_idx}.json')
        memory.save_skills(snapshot_path)

        wall_time = time.time() - t_start
        round_record = {
            'round': round_idx,
            'split': args.split,
            'enumerate': args.enumerate,
            'inject_style': args.inject_style,
            'doc_file': args.doc_file,
            'rollout_games': len(records),
            'rollout_temp': args.rollout_temp,
            'n_api_error_episodes': sum(1 for r in records if r['n_api_errors'] > 0),
            'overall_success': overall_success,
            'per_task_type': per_type,
            'weak_threshold': args.weak_threshold,
            'weak_task_types': weak_types,
            'target_task_types': target_types,
            'max_ops_per_round': args.max_ops_per_round,
            'candidates': candidates,
            'writer_silence': writer_silence,
            'grounding': args.grounding,
            'grounding_strictness': args.grounding_strictness,
            'grounding_filtered': round_grounding_filtered,
            'ops': ops,
            'allow_revisions': args.allow_revisions,
            'gate': gate_record,
            'accepted_task_types': [c['task_type'] for c in accepted_candidates],
            'rejection_memory': copy.deepcopy(last_rejection),
            'library_size_before': library_size_before,
            'library_size_after': memory.get_skill_count(),
            'library_snapshot': snapshot_path,
            'wall_time_sec': wall_time,
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(round_record) + '\n')
        print(f"[bootstrap] round {round_idx} done in {wall_time:.1f}s | "
              f"library {library_size_before['total']} -> "
              f"{memory.get_skill_count()['total']} skills | log -> {log_path}",
              flush=True)

    print(f"\n[bootstrap] Finished {args.rounds} round(s). "
          f"Final library: {os.path.join(out_dir, f'library_round{args.rounds}.json')}",
          flush=True)


if __name__ == '__main__':
    main()
