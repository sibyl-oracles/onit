# Reference

External sources behind OnIt's design. Per-document citation lists live with the
documents that use them — see [`SELF_IMPROVEMENT.md`](SELF_IMPROVEMENT.md) §Sources for
the self-improvement literature and [`NOOA_onit_recommendations.md`](NOOA_onit_recommendations.md)
§References for the harness-capability papers. This file collects the primary ones.

---

## The harness, not the model

**[Nvidia just showed that the harness, not the AI model, is now the real hero](https://techcrunch.com/2026/08/21/nvidia-just-showed-that-the-harness-not-the-ai-model-is-now-the-real-hero/)**
— Julie Bort, TechCrunch, August 21, 2026.

Reports NVIDIA research in which Claude Opus 5 scored **100%** on the ARC-AGI-3
interactive reasoning benchmark when driven by a custom harness, against **30%** for the
same model without one; OpenAI models are reported below 10% on the same benchmark. The
harness NVIDIA built — called **Agentic Variation Operators (AVO)** — supplies memory
management, context handling, tool access, and a *supervisor* that intervenes when the
agent stalls or works an unproductive path.

> "The harness is what makes a model an agent: It handles memory, context, and feedback."

Three claims from the piece bear directly on OnIt:

- Model choice is secondary to harness quality on long-horizon tasks.
- Harness design can double operating cost — the loop, not the weights, sets the bill.
- Open harnesses give operators control that a proprietary model endpoint does not.

**Relevance.** This is the external case for the work already recorded in
[`HARNESS_CAPABILITIES.md`](HARNESS_CAPABILITIES.md): run-state budgeting, the result
store, harness tools, early stopping, and answer verification are OnIt's versions of the
same wager. AVO's *supervisor* has no direct OnIt counterpart today; the closest pieces
are early stopping and answer verification, which act after a step rather than steering
during one.

**Caveat.** These are figures as reported by TechCrunch, not results reproduced here, and
a single-benchmark result on puzzle games generalises poorly. Treat the direction as
signal and the numbers as journalism until the primary NVIDIA publication is available.

---

## Agent harness research

| Source | Use in OnIt |
|---|---|
| [Six Agent Harness Capabilities for Higher Model Performance](https://developer.nvidia.com/blog/six-agent-harness-capabilities-for-higher-model-performance/) (NVIDIA, NOOA framework) | The six-capability scorecard driving [`HARNESS_CAPABILITIES.md`](HARNESS_CAPABILITIES.md) |
| [Awesome-Self-Improving-Agents](https://github.com/FrontisAI/Awesome-Self-Improving-Agents) | Survey feeding [`SELF_IMPROVEMENT.md`](SELF_IMPROVEMENT.md) |

## Protocols

| Source | Use in OnIt |
|---|---|
| [Model Context Protocol](https://modelcontextprotocol.io/) | Every tool server (`src/mcp/`) |
| [A2A Protocol](https://a2a-protocol.org/) | `onit serve a2a`, agent-to-agent transport |

## Model serving

| Source | Use in OnIt |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | Primary self-hosted endpoint |
| [Ollama](https://ollama.com) · [ollama-python](https://github.com/ollama/ollama-python) | Local and cloud fallback endpoints |
| [MLX LM](https://github.com/ml-explore/mlx-lm) | Apple-silicon local serving |
| [OpenRouter](https://openrouter.ai/) | Hosted multi-model endpoint |
| [Hugging Face](https://huggingface.co) | Model and weight distribution |

See [`MODEL_SERVING.md`](MODEL_SERVING.md).

## Tools and data

| Source | Use in OnIt |
|---|---|
| [Mistral Search Toolkit](https://mistral.ai/news/search-toolkit/) | Parse → chunk → embed → retrieve design of [`LOCAL_SEARCH.md`](LOCAL_SEARCH.md) |
| [Ollama web search](https://ollama.com/blog/web-search) | Web search tool backend |
| [NVIDIA NemotronLabs VoiceChat](https://github.com/NVIDIA-NeMo/Speech/tree/nemotron-labs-voicechat) | Speech-to-speech backend for [`VOICE.md`](VOICE.md) |

## Deployment

| Source | Use in OnIt |
|---|---|
| [Docker](https://docs.docker.com/get-docker/) | `--container` mode, [`DOCKER.md`](DOCKER.md) |
| [Caddy](https://caddyserver.com) · [Let's Encrypt](https://letsencrypt.org) · [Certbot](https://certbot.eff.org) | TLS termination, [`HTTPS_DEPLOYMENT.md`](HTTPS_DEPLOYMENT.md) |
| [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) · [Tailscale](https://tailscale.com/download) | Public webhook exposure, [`GATEWAY_QUICK_START.md`](GATEWAY_QUICK_START.md) |
| [Google OAuth 2.0](https://developers.google.com/identity/protocols/oauth2/web-server) | [`WEB_AUTHENTICATION.md`](WEB_AUTHENTICATION.md), [`OAUTH_SETUP_QUICK_START.md`](OAUTH_SETUP_QUICK_START.md) |
