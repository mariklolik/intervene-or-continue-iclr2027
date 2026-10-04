import re
import numpy as np
import pandas as pd
_FAIL_OBS_PAT = re.compile("(no known action matches|nothing happens|that's not something you|you can't|cannot|not a valid|no such thing|i don't|invalid|there is no way to do that|unknown action)", re.IGNORECASE)
_VERB_GROUPS: dict[str, tuple[str, ...]] = {'move': ('go to', 'go ', 'teleport', 'move to'), 'take': ('take', 'pick up', 'pick '), 'put': ('put', 'move ', 'place'), 'open': ('open',), 'close': ('close',), 'toggle': ('use', 'activate', 'deactivate', 'turn on', 'turn off', 'toggle'), 'look': ('look', 'examine', 'inspect'), 'read': ('read',), 'focus': ('focus',), 'manip': ('clean', 'heat', 'cool', 'slice', 'mix', 'pour', 'connect', 'wait', 'dunk')}
_TOKEN_PAT = re.compile('[a-z0-9]+')

def _tok(text: str) -> list[str]:
    return _TOKEN_PAT.findall(text.lower())

def _safe_bool(value, default: bool=False) -> bool:
    if value is None:
        return default
    return bool(value)

def prefix_features(episode: dict, k: int) -> dict[str, float]:
    steps = episode['steps'][:k]
    assert len(steps) == k, f'prefix requires {k} steps, episode has {len(steps)}'
    actions = [s.get('parsed_action') or '' for s in steps]
    observations = [s.get('observation') or '' for s in steps]
    rewards = np.array([float(s.get('reward') or 0.0) for s in steps])
    state_changed = np.array([_safe_bool(s.get('state_changed'), True) for s in steps])
    reversible = np.array([_safe_bool(s.get('reversible'), True) for s in steps])
    invalid = np.array([_safe_bool(s.get('invalid_output'), False) for s in steps])
    retries = np.array([float(s.get('parse_retries') or 0) for s in steps])
    admissible_n = np.array([float(s.get('admissible_n') or 0) for s in steps])
    act_adm = [s.get('action_admissible') for s in steps]
    obs_hash = [s.get('obs_hash') or '' for s in steps]
    prompt_len = np.array([float(s.get('prompt_len') or 0) for s in steps])
    out_tok = np.array([float((s.get('usage') or {}).get('output_tokens') or 0) for s in steps])
    feats: dict[str, float] = {}
    feats['steps_so_far'] = float(k)
    feats['log_steps_so_far'] = float(np.log1p(k))
    feats['env_scienceworld'] = 1.0 if episode['env'] == 'scienceworld' else 0.0
    cum_reward = float(rewards.sum())
    feats['cum_reward'] = cum_reward
    feats['any_reward'] = 1.0 if cum_reward > 0 else 0.0
    feats['reward_rate'] = cum_reward / k
    nz = np.flatnonzero(rewards > 0)
    feats['steps_since_reward'] = float(k - 1 - nz[-1]) if nz.size else float(k)
    feats['frac_steps_rewarded'] = float((rewards > 0).mean())
    feats['n_invalid_output'] = float(invalid.sum())
    feats['rate_invalid_output'] = float(invalid.mean())
    feats['n_parse_retries'] = float(retries.sum())
    feats['n_state_unchanged'] = float((~state_changed).sum())
    feats['rate_state_unchanged'] = float((~state_changed).mean())
    trail = 0
    for flag in reversed(state_changed.tolist()):
        if flag:
            break
        trail += 1
    feats['stall_state_unchanged_run'] = float(trail)
    feats['n_irreversible'] = float((~reversible).sum())
    feats['last_irreversible'] = 0.0 if reversible[-1] else 1.0
    feats['mean_admissible_n'] = float(admissible_n.mean())
    feats['last_admissible_n'] = float(admissible_n[-1])
    known = [a for a in act_adm if a is not None]
    feats['action_admissible_rate'] = float(np.mean([bool(a) for a in known])) if known else -1.0
    feats['n_action_inadmissible'] = float(sum((1 for a in known if a is False)))
    uniq_actions = len(set(actions))
    feats['distinct_action_ratio'] = uniq_actions / k
    counts = pd.Series(actions).value_counts()
    feats['max_action_repeat'] = float(counts.iloc[0]) if len(counts) else 0.0
    feats['n_repeated_actions'] = float(k - uniq_actions)
    feats['n_consecutive_repeats'] = float(sum((1 for i in range(1, k) if actions[i] == actions[i - 1])))
    feats['last_action_seen_before'] = 1.0 if actions[-1] in set(actions[:-1]) else 0.0
    uniq_obs = len(set(obs_hash))
    feats['distinct_obs_ratio'] = uniq_obs / k
    feats['n_repeated_obs'] = float(k - uniq_obs)
    feats['last_obs_seen_before'] = 1.0 if obs_hash[-1] in set(obs_hash[:-1]) else 0.0
    failed = np.array([bool(_FAIL_OBS_PAT.search(o)) for o in observations])
    feats['n_failed_obs'] = float(failed.sum())
    feats['rate_failed_obs'] = float(failed.mean())
    feats['last_obs_failed'] = 1.0 if failed[-1] else 0.0
    trail_f = 0
    for flag in reversed(failed.tolist()):
        if not flag:
            break
        trail_f += 1
    feats['stall_failed_obs_run'] = float(trail_f)
    seen: set[str] = set()
    novel_counts: list[float] = []
    total_tokens = 0
    for obs in observations:
        toks = _tok(obs)
        total_tokens += len(toks)
        if toks:
            novel_counts.append(sum((1 for t in toks if t not in seen)) / len(toks))
        else:
            novel_counts.append(0.0)
        seen.update(toks)
    feats['novel_token_frac_last'] = float(novel_counts[-1])
    feats['novel_token_frac_mean'] = float(np.mean(novel_counts))
    feats['novel_token_frac_last3'] = float(np.mean(novel_counts[-3:]))
    feats['vocab_size_seen'] = float(len(seen))
    feats['vocab_per_step'] = len(seen) / k
    feats['token_repeat_ratio'] = float(len(seen) / total_tokens) if total_tokens else 0.0
    feats['last_prompt_len'] = float(prompt_len[-1])
    feats['prompt_len_per_step'] = float(prompt_len[-1] / k)
    feats['mean_output_tokens'] = float(out_tok.mean())
    feats['last_output_tokens'] = float(out_tok[-1])
    feats['mean_obs_len'] = float(np.mean([len(o) for o in observations]))
    feats['last_obs_len'] = float(len(observations[-1]))
    lowered = [a.lower().strip() for a in actions]
    assigned = np.zeros(k, dtype=bool)
    for group, prefixes in _VERB_GROUPS.items():
        hit = np.array([any((a.startswith(p) or a.split(' ')[0] == p.strip() for p in prefixes)) for a in lowered])
        hit &= ~assigned
        assigned |= hit
        feats[f'act_frac_{group}'] = float(hit.mean())
    feats['act_frac_other'] = float((~assigned).mean())
    return feats

def recent_text(episode: dict, k: int, n_recent: int=3) -> str:
    steps = episode['steps'][:k][-n_recent:]
    parts: list[str] = []
    for s in steps:
        parts.append(s.get('parsed_action') or '')
        parts.append(s.get('observation') or '')
    return ' '.join(parts)
