# Search MAS Tree Smoke

This example is a manual reproduction entrypoint for the real Search MAS tree-rollout smoke.

It is not intended to run in CI. The smoke needs:

- GPUs with enough free memory for 3 independent rollout/train workers
- local model paths under `/data1/lll/models`
- a live retrieval service

## Verified Baseline

The current smoke config is:

- Hydra config: `orchrl/config/search/search_mas_tree_real_smoke.yaml`
- specialization: `full`
- model: `Qwen3-0.6B`
- policies: one per agent
- `training.max_prompt_length=2048`
- `training.max_response_length=1024`
- `application.max_turns=4`

## 1. Start Retrieval Service

On this server, one verified command is:

```bash
cd /home/cxb/OrchRL

python3 -m pip install faiss-cpu

env CUDA_VISIBLE_DEVICES=7 python3 \
  examples/mas_app/search/scripts/retrieval_server.py \
  --index_path /data1/lll/datasets/wiki-18/e5_Flat.index \
  --corpus_path /data1/lll/datasets/wiki-18/wiki-18.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model /data1/lll/models/e5-base-v2 \
  --port 8010
```

Optional health check:

```bash
curl -fsS -X POST http://127.0.0.1:8010/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"healthcheck","topk":1}'
```

## 2. Preflight

```bash
cd /home/cxb/OrchRL

examples/search_mas_tree/run.sh \
  --check-only \
  --skip-retrieval-healthcheck
```

## 3. Run The Smoke

```bash
cd /home/cxb/OrchRL

examples/search_mas_tree/run.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
RETRIEVAL_SERVICE_URL=http://127.0.0.1:8010/retrieve \
LOG_PATH=/home/cxb/OrchRL/logs/search_mas_tree_real_smoke.log \
examples/search_mas_tree/run.sh
```

## 4. Success Signals

Inspect `logs/search_mas_tree_real_smoke.log`. A valid smoke should show:

- `step 0 started`
- `step:1 - ... training/global_step:0.000`
- `step 1 started`
- `Cleanup completed`

The current real Search MAS trace can legitimately skip a policy in a step. A log line like this is acceptable:

- `MATE rollout produced partial policy batches; missing=['searcher_model'], available=['verifier_model', 'answerer_model']`
