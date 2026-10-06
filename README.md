<div align="center">

# VirusCascade: Hijacking Collaborative Reflection in LLM-Powered Recommender Agents

**NDSS 2027 · Oral**

[Yurong Hao](https://yuronghaoa.github.io/yuronghaoA/)†, [Wen Zhou](https://scholar.google.com/citations?hl=zh-CN&user=_NHZPoYAAAAJ&view_op=list_works&sortby=pubdate), [Guowei Guan](https://scholar.google.com/citations?user=cMJEfBsAAAAJ&hl=zh-CN), [Tiantong Wu](https://sites.google.com/view/ttwu), [Fuyao Zhang](https://fyzhang1.github.io/#home), [Wei Yang Bryan Lim](https://sites.google.com/view/wyb/)

College of Computing and Data Science, Nanyang Technological University

[![arXiv](https://img.shields.io/badge/arXiv-2609.38270-b31b1b.svg)](https://arxiv.org/abs/2609.38270)
[![PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](https://arxiv.org/pdf/2609.38270)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://yuronghaoa.github.io/VirusCascade-website/)
[![NDSS 2027](https://img.shields.io/badge/NDSS-2027-green.svg)](https://yuronghaoa.github.io/VirusCascade-website/)

<img src="https://yuronghaoa.github.io/VirusCascade-website/static/images/fig3_overview.png" width="90%" alt="VirusCascade overview">

</div>

## 📌 Overview

**LLM-powered agentic recommender systems (LLM-ARS)** model every user and every item as an LLM agent with natural-language memory. After each interaction, agents *reflect* on what happened, write the conclusion back into memory, and share it with the agents they interact with. This **collaborative reflection** brings strong personalization and interpretability, but it also means that what one agent writes is read, reused, and reflected on by others.

**VirusCascade** is a black-box targeted promotion attack that turns collaborative reflection into an attack surface. It is built on two observations:

- **Reflective persistence.** Once admitted, an injected claim stays active across reflection–writeback cycles. Admitted users keep reusing the claim at 78–82% per round, versus 2.0% without injection.
- **Cross-agent propagation.** A local memory update reaches agents that were never directly modified, decaying with hop distance in the user–item graph (33.6% adoption at hop 1, 25.8% at hop 2). Admitted benign users grow from 2% to 38% in five rounds with no further attacker action.

Persistence decides *what* to inject; propagation decides *where* to inject it. VirusCascade exploits both:

1. **Semantic injection** (← persistence): craft the target item's profile so that interactions with it are naturally rationalized as coherent preference evidence during reflection.
2. **Structural injection** (← propagation): construct behavior trajectories that guide attacker-controlled users along smooth, propagation-oriented paths through high-degree items.
3. **Reflection-driven amplification**: after injection, all further amplification is carried out by the victim system's own collaborative reflection.

## 🏆 Highlights

| | |
| --- | --- |
| **0.384** mean E@20 | vs. 0.199 for the best baseline (+0.185 absolute) |
| **1%** malicious users | are enough (5 of 500), and the effect persists for 20 rounds |
| **Stealthy** | perplexity 32.22, within the range of original profiles (10.31–90.02); human flag rate 21.9% vs. 19.3% for originals |
| **Utility-preserving** | recommendation quality (H@K / N@K) stays at the clean level |
| **Transferable** | 4 datasets, 3 victim architectures; works with LLaMA-3, GPT-4o and Gemini-2.5 as auxiliary models |

E@20 (targeted exposure, higher is better) against the strongest baseline in each setting:

| Victim | CDs & Vinyl (best baseline) | CDs & Vinyl (**VirusCascade**) | Movies & TV (best baseline) | Movies & TV (**VirusCascade**) |
| --- | --- | --- | --- | --- |
| AgentCF | 0.091 | **0.374** | 0.150 | **0.240** |
| AgentSEQ | 0.071 | **0.465** | 0.270 | **0.300** |
| AgentRAG | 0.380 | **0.545** | 0.230 | **0.380** |

See the [paper](https://arxiv.org/abs/2609.38270) for full comparisons against interaction-level, text-level, and memory-level baselines, ablations, and defenses (ONION, Fraudar, TrustRAG, A-MemGuard).

## 🧩 From Paper to Code

The code lives in the `agentpoisoning` package. For each target item, `AgentPoisoningRunner` runs the two injections:

**Semantic injection → `_generate_ugc` (config section `induction`)**

- **Hub selection.** The most popular items in the training interactions are taken as hub items (`hub_count`).
- **Motif extraction.** An LLM condenses the purchase drivers shared by the hub items into a short motif (`prompts/induction_motif.txt`).
- **Contrastive decoding.** The target profile is generated under an expert prompt and an amateur prompt; token scores are `l_cd = logp_expert − λ · logp_amateur` (`contrastive_lambda`), filtered by a plausibility mask.
- **Profile refinement.** The result is rewritten into several lightly edited variants for fluency and diversity (`refine_ugc`, `prompts/ugc_refine.txt`).

**Structural injection → `_plan_trigger_sequences` (config section `trigger`)**

- **Anchors and bridges.** Top hub items act as anchors (`num_anchors`); the remaining hub items form the bridge pool.
- **Transition cost.** `cost = α · embedding_distance + (1 − α) · llm_naturalness` (`cost_alpha`, `embedding_model`, `prompts/trigger_naturalness.txt`).
- **Path planning.** A fixed-length shortest path `anchor → bridges → target` is searched for each anchor; each path becomes one malicious user's interaction sequence.

Item IDs are then mapped to AgentCF IDs / ASINs and written to a single JSON file, ready to be injected into the victim simulator.

## 📁 Repository Structure

```
VirusCascade/
├── run_agentpoisoning.py        # Entry point (CLI)
├── configs/
│   └── config.json              # Example configuration
├── agentpoisoning/
│   ├── attack.py                # AgentPoisoningRunner: semantic + structural injection
│   ├── config.py                # Dataclass-based configuration
│   ├── dataset.py               # Item / interaction loading, popularity statistics
│   ├── llm.py                   # API client (OpenAI / Qwen / Gemini / Claude / DeepSeek)
│   ├── llama_llm.py             # Local HuggingFace client (LLaMA)
│   ├── prompts/                 # Prompt templates
│   └── utils/                   # Text helpers
└── dataset-V1/                  # Amazon subsets: Automotive, CDs, Instruments, Movies
```

## ⚙️ Installation

Python ≥ 3.9 is recommended.

```bash
git clone https://github.com/yuronghaoA/VirusCascade.git
cd VirusCascade
pip install openai requests
# Only needed for the local LLaMA backend
pip install torch transformers
```

## 📊 Data

`dataset-V1/` contains the four Amazon domains used in the paper (CDs & Vinyl, Movies & TV, Automotive, Musical Instruments), in RecBole-style atomic files:

| File | Content |
| --- | --- |
| `<name>.item` | `item_id`, `title`, `category` |
| `<name>.pretrained_item` | `item_id`, LLM-generated item description |
| `<name>.train.inter` / `.valid.inter` / `.test.inter` | `user_id`, history `item_id_list`, next `item_id` |
| `<name>.random` | Candidate items per user |

**ID mapping.** The runner also requires a `rectext_mapping.json` that maps item IDs to AgentCF IDs and ASINs. It is resolved relative to the repository's parent directory:

```
<parent_dir>/4AgentCF/attackRes/gpt-4o-mini/<dataset>/0_cache/rectext_mapping.json
```

Use `--agentcf-dataset` if the mapping folder name differs from `dataset_name`.

## 🔧 Configuration

All settings live in a JSON file (see `configs/config.json`). Before running, update the paths in `dataset.data_dir` / `dataset.item_file` and fill in your API keys.

| Section | Key | Description |
| --- | --- | --- |
| `dataset` | `data_dir`, `dataset_name`, `domain_label` | Dataset location, name, and the domain word used in prompts (e.g. `CD`) |
| | `target_item_ids` | Items to promote; if empty, the least popular item is chosen |
| `attack` | `dry_run` | `true` only prepares context and prompts without calling the LLM |
| | `save_partial` | Write results after each target item |
| `induction` | `hub_count` | Number of popular hub items |
| | `contrastive_lambda` | λ in contrastive decoding |
| | `max_words` | Word limit of the generated profile |
| | `refine_ugc` | Whether to rewrite the profile into variants |
| `trigger` | `num_anchors` | Number of anchors (= malicious users per target) |
| | `cost_alpha` | α balancing embedding distance vs. naturalness; the paper finds **α = 0.7–0.9** gives the best trade-off |
| | `embedding_model` | Embedding model for item distances |
| | `use_llm_naturalness` | Score transitions with an LLM |
| `llm` | `backend` | `openai`, `qwen`, `gemini`, `claude`, `deepseek`, or `llama` |
| | `model`, `temperature`, `max_completion_tokens` | Generation settings for the auxiliary attack model |
| | `api_key_list` | Keys for the main backend |
| | `auxiliary_openai_api_key_list` | OpenAI keys for logprobs / embeddings when the main backend lacks them |

API keys can also be supplied through environment variables:

| Backend | Variable |
| --- | --- |
| OpenAI | `OPENAI_API_KEY` (optional `OPENAI_API_BASE`) |
| Qwen | `QWEN_API_KEY` or `DASHSCOPE_API_KEY` |
| Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| Claude | `ANTHROPIC_API_KEY` |
| DeepSeek | `DASHSCOPE_API_KEY` or `DEEPSEEK_API_KEY` |

## 🚀 Usage

Run with the config file:

```bash
python run_agentpoisoning.py --config configs/config.json --execute-llm
```

Without `--execute-llm`, the run stays in dry-run mode.

Command-line arguments override the config:

```bash
python run_agentpoisoning.py \
    --config configs/config.json \
    --data-dir dataset-V1/CDs \
    --dataset CDs \
    --target-items 123,456,789 \
    --contrastive-lambda 0.7 \
    --cost-alpha 0.7 \
    --output outputs \
    --execute-llm
```

| Argument | Description |
| --- | --- |
| `--config` | Path to the JSON config |
| `--data-dir` / `--dataset` | Override dataset directory / name |
| `--agentcf-dataset` | Dataset name used to locate `rectext_mapping.json` |
| `--target-items` | Comma-separated target item IDs |
| `--contrastive-lambda` | Override `induction.contrastive_lambda` |
| `--cost-alpha` | Override `trigger.cost_alpha` |
| `--output` | Output directory |
| `--execute-llm` | Actually call the LLM (disables dry run) |

## 📤 Output

Results are saved to:

```
outputs/agentpoisoning-{λ}-{α}.{dataset}-{model}.json
```

keyed by target item ID:

```json
{
  "123": {
    "attack": "agentpoisoning",
    "dataset": "CDs",
    "target_item_id": 123,
    "agentcf_id": 45,
    "asin": "B000XXXXXX",
    "target_item_title": "...",
    "original_prompt": "...",
    "attacked_prompt": "...",
    "ugc": {
      "text": "...",
      "texts": ["...", "...", "..."],
      "motif_text": "...",
      "hub_item_ids": [...],
      "raw_expert_review": "...",
      "raw_amateur_review": "..."
    },
    "trigger": {
      "anchor_item_ids": [...],
      "paths": [[...]],
      "fake_users": [{"user_id": "...", "sequence": ["ASIN_1", "ASIN_2", "..."]}]
    }
  }
}
```

- `attacked_prompt` is the semantic injection: it replaces the target item's profile.
- `trigger.fake_users` is the structural injection: each entry is a malicious user's interaction sequence ending with the target item.

Both are injected into the victim simulator (AgentCF / AgentSEQ / AgentRAG); amplification then happens through the victim's own collaborative reflection.

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{hao2027viruscascade,
  title     = {VirusCascade: Hijacking Collaborative Reflection in LLM-Powered Recommender Agents},
  author    = {Hao, Yurong and Zhou, Wen and Guan, Guowei and Wu, Tiantong and Zhang, Fuyao and Lim, Wei Yang Bryan},
  booktitle = {Network and Distributed System Security (NDSS) Symposium},
  year      = {2027}
}
```

## ⚖️ Ethics Statement

All experiments run in a sandboxed local simulation on public, anonymized Amazon datasets; no real platform, production API, or user is targeted. Findings were disclosed to the affected open-source framework developers before publication. This code is released to expose an overlooked attack surface in LLM-powered agentic recommender systems and to support research on robust defenses. Please do not use it against real-world systems.
