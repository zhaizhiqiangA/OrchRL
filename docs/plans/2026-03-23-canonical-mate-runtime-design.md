# Canonical MATE Runtime Design

**Date:** 2026-03-23

## Goal

Unify the outer OrchRL trajectory engine with the inner canonical-runtime interface so that training uses a single tokenization path end-to-end:

`prompt_loader -> monitor local render(prompt_ids) -> VERL direct generate(prompt_ids) -> token_ids/logprobs -> trajectory -> trainer batch`

The outer training path must stop re-tokenizing `turn.messages` during batch construction.

## Current State

The outer repo already has a working direct-actor training path, but it is split across:

- a simplified trajectory engine interface
- a dedicated direct backend implementation
- trainer-side fallback tokenization in `mate_dataproto_adapter`

The inner repo has a cleaner canonical-runtime design:

- `ModelRequest` carries `prompt_ids`
- `ModelMonitor` locally renders canonical prompt ids
- `VerlBackend` sends prompt ids directly to `server_manager.generate(...)`
- runtime validation and drift diagnostics are first-class

However, the inner trainer still instantiates `VLLMBackend` and still re-tokenizes `turn.messages` in `mate_dataproto_adapter`, so the architecture is not wired end-to-end yet.

## Target Architecture

The outer repo will adopt the inner trajectory-engine interface model and connect it to the already-running outer training path.

### Runtime Contract

`ModelRequest` must carry:

- `messages`
- `generation_params`
- `prompt_ids`
- `render_fingerprint`
- `sampling_fingerprint`

`ModelResponse` must carry:

- `content`
- `token_ids`
- `logprobs`
- `finish_reason`
- `prompt_ids`
- optional `routed_experts`
- `runtime_metadata`

Each recorded turn in the trajectory must preserve:

- `prompt_ids`
- `token_ids`
- `logprobs`
- replay metadata for tree rollouts

### Tokenization Ownership

Canonical prompt tokenization happens exactly once per request in `ModelMonitor` via a local `ChatRenderer`.

`VerlBackend` consumes the rendered `prompt_ids` directly and passes them to VERL/vLLM actor handles using `generate(prompt_ids=...)`.

Trainer code must treat `turn.prompt_ids` as the source of truth. Missing prompt ids are a hard error on the canonical path.

### Backend Model

The trajectory engine exposes two backends:

- `VLLMBackend`: HTTP OpenAI-compatible compatibility path
- `VerlBackend`: direct token-in/token-out path used by training

The outer trainer must prefer `VerlBackend` whenever `server_handle_dict` and `tokenizer_dict` are available.

## Component Changes

### `orchrl/agent_trajectory_engine`

The outer package will be reshaped to match the inner interface:

- expand `datatypes.py`
- merge backend logic into `backend.py`
- add `_support/renderer.py`
- add `_support/validator.py`
- add `_support/diagnostics.py`
- add `_support/collector.py`
- move the HTTP interception logic to canonical-monitor behavior

The old split between `backend.py`, `actor_backend.py`, and `gateway.py` becomes unnecessary after the unification.

### `mate_rollout_adapter`

The rollout adapter becomes the integration point between trainer state and canonical runtime:

- build canonical `AgentPipeConfig`
- construct a renderer from per-policy tokenizers
- build `VerlBackend` from `server_handle_dict`
- preserve existing tree and parallel rollout topology

### `mate_dataproto_adapter`

The adapter must stop calling `_tokenize_messages(...)` on the training side. It should:

- normalize and truncate `turn.prompt_ids`
- fail fast if canonical prompt ids are missing
- keep using recorded response ids from rollout

This removes the remaining double-tokenization risk.

## Tree and Replay Semantics

Tree rollout topology does not change. Branching still uses replay caches and branch metadata, but the canonical fields must survive:

- `prompt_ids`
- `token_ids`
- `logprobs`
- `replayed`
- `branch_phase`

This ensures tree-mode PPO sees the exact runtime prompt ids that generated each branch.

## Drift Detection

For every request, the monitor writes a drift artifact containing:

- original `messages`
- runtime `prompt_ids`
- locally re-rendered `prompt_ids`
- response ids
- response logprobs
- render fingerprint
- sampling fingerprint
- mismatch flag

On the canonical path, any missing prompt ids or malformed response ids/logprobs are validation failures.

## Failure Policy

The canonical path is strict:

- missing `prompt_ids` in request or trajectory is an error
- empty `response.token_ids` is an error
- token/logprob length mismatch is an error
- trainer fallback re-tokenization is removed

The goal is to fail early instead of silently drifting.

## Verification Strategy

Verification will happen at three levels.

### Unit Level

- backend tests for `VerlBackend`
- monitor tests for local render and drift artifact creation
- datatypes/collector tests for metadata preservation
- trainer adapter tests that missing `turn.prompt_ids` fails

### Integration Level

- one-step smoke rollout validating every turn has `prompt_ids/token_ids/logprobs`
- trainer batch assertions showing `prompts` come from recorded prompt ids

### End-to-End Level

- run `bash scripts/run_search_mas_train_e2e.sh`
- trace `prompt_loader -> monitor render -> VerlBackend.generate -> vLLM token ids -> dataproto -> old_log_prob -> adv -> update_actor`
- confirm one sampled turn's `turn.prompt_ids` exactly matches the trainer batch prompt tensor

## Non-Goals

- changing reward semantics
- changing tree/parallel rollout scheduling semantics
- changing PPO optimization logic

The refactor is strictly about unifying the runtime interface and removing duplicate tokenization paths.
