//! llama.cpp (`llama-server`) adapter.
//!
//! llama-server exposes its telemetry at `/metrics` in Prometheus text
//! exposition format, but only when started with the `--metrics` flag —
//! without it the endpoint answers with a JSON error, which the parser
//! rejects, so the engine keeps its status but reads `metrics: null`.
//!
//! Metric names were verified against the llama.cpp sources
//! (`tools/server/`, releases b4000 through master): every metric is
//! exposed under the reserved `llamacpp:` prefix, which
//! `prometheus::parse_prometheus_text` normalizes to `llamacpp_` before
//! parsing.
//!
//! The metric set is deliberately smaller than vLLM's:
//!
//! * **No per-request histograms** — there is no TTFT, no end-to-end
//!   latency, and no per-slot per-token timing. Every histogram-derived
//!   field of `EngineMetrics` (TTFT, E2E, percentiles, goodput, raw
//!   buckets) is `None` for this engine.
//! * **No total-request counter** — `total_requests` is `None`.
//! * **Engine-computed windowed throughput** — `llamacpp:prompt_tokens_seconds`
//!   and `llamacpp:predicted_tokens_seconds` are gauges the server averages
//!   over the interval between two scrapes, so they are read directly
//!   instead of diffing cumulative counters between polls.
//! * **Per-token decode time is derivable** — the cumulative
//!   `tokens_predicted_seconds_total` / `tokens_predicted_total` pair gives
//!   the engine's lifetime average time per generated token (TPOT ≈ ITL)
//!   and its decode throughput, so those fields are populated from the
//!   ratio rather than left blank.
//! * **Prompt cache and speculative decoding are reported** on current
//!   releases (`prompt_tokens_cached_total`, `spec_decode_*`), mapped onto
//!   the shared prefix-cache and spec-decode fields of the contract.
//!
//! There is no per-request histogram, so there is nothing for the
//! `warmup` state machine to screen: `warming_up` is always `false`.

use super::prometheus::{parse_prometheus_text, ParsedMetrics};
use super::vllm::{
    classify_models_error_status, normalize_model_id, spec_acceptance_rate,
    spec_mean_acceptance_length, OpenAIModelsResponse,
};
use super::{
    EngineAdapter, EngineMetrics, EngineStatus, EngineType, ModelInfo, ModelMetadataError,
    ModelResolution,
};
use async_trait::async_trait;
use std::time::Duration;
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// Metric names (verified against llama.cpp `tools/server/`, master + b10000)
// ---------------------------------------------------------------------------

/// Cumulative prompt tokens processed (excluding cached tokens).
const PROMPT_TOKENS_TOTAL: &str = "llamacpp_prompt_tokens_total";
/// Cumulative prompt tokens reused from the prompt cache.
const PROMPT_TOKENS_CACHED_TOTAL: &str = "llamacpp_prompt_tokens_cached_total";
/// Cumulative seconds spent on prompt processing (prefill).
const PROMPT_SECONDS_TOTAL: &str = "llamacpp_prompt_seconds_total";
/// Cumulative tokens generated (decoded).
const TOKENS_PREDICTED_TOTAL: &str = "llamacpp_tokens_predicted_total";
/// Cumulative seconds spent on token generation (decode).
const TOKENS_PREDICTED_SECONDS_TOTAL: &str = "llamacpp_tokens_predicted_seconds_total";
/// Gauge: average prompt throughput over the window between two scrapes.
const PROMPT_TOKENS_SECONDS: &str = "llamacpp_prompt_tokens_seconds";
/// Gauge: average generation throughput over the window between two scrapes.
const PREDICTED_TOKENS_SECONDS: &str = "llamacpp_predicted_tokens_seconds";
/// Gauge: requests currently being processed.
const REQUESTS_PROCESSING: &str = "llamacpp_requests_processing";
/// Gauge: requests deferred (queued) waiting for a slot.
const REQUESTS_DEFERRED: &str = "llamacpp_requests_deferred";
/// Gauge: average number of busy slots per decode call — llama.cpp's
/// batch-size proxy.
const N_BUSY_SLOTS_PER_DECODE: &str = "llamacpp_n_busy_slots_per_decode";
/// Gauge (older releases): KV cache usage ratio in [0, 1].
const KV_CACHE_USAGE_RATIO: &str = "llamacpp_kv_cache_usage_ratio";
/// Cumulative speculatively-drafted tokens.
const SPEC_DECODE_DRAFT_TOKENS_TOTAL: &str = "llamacpp_spec_decode_num_draft_tokens_total";
/// Cumulative draft tokens that passed verification.
const SPEC_DECODE_ACCEPTED_TOKENS_TOTAL: &str = "llamacpp_spec_decode_num_accepted_tokens_total";
/// Cumulative speculative-decoding draft attempts.
const SPEC_DECODE_DRAFTS_TOTAL: &str = "llamacpp_spec_decode_num_drafts_total";
/// Cumulative accepted tokens per draft position, labeled by `position`.
const SPEC_DECODE_ACCEPTED_PER_POS_TOTAL: &str =
    "llamacpp_spec_decode_num_accepted_tokens_per_pos_total";
/// Cumulative number of llama_decode() calls.
const N_DECODE_TOTAL: &str = "llamacpp_n_decode_total";
/// Largest observed sequence length (prompt + generation).
const N_TOKENS_MAX: &str = "llamacpp_n_tokens_max";

/// Map one parsed `/metrics` scrape onto the shared `EngineMetrics`
/// contract.
///
/// Pure with respect to the adapter: the previous spec-decode counter
/// reading and the two throughput accumulators are passed in and returned,
/// which keeps the entire mapping — including its deliberate `None`s —
/// unit-testable without HTTP.
fn map_metrics(
    parsed: &ParsedMetrics,
    prev_spec_decode: Option<(f64, f64)>,
    avg_generation: &mut (f64, u64),
    avg_prompt: &mut (f64, u64),
) -> (EngineMetrics, Option<(f64, f64)>) {
    let prompt_total = parsed.counters.get(PROMPT_TOKENS_TOTAL).copied();
    let prompt_cached = parsed.counters.get(PROMPT_TOKENS_CACHED_TOTAL).copied();
    let prompt_seconds = parsed.counters.get(PROMPT_SECONDS_TOTAL).copied();
    let generation_total = parsed.counters.get(TOKENS_PREDICTED_TOTAL).copied();
    let generation_seconds = parsed.counters.get(TOKENS_PREDICTED_SECONDS_TOTAL).copied();

    // The engine averages its own throughput over the window between two
    // scrapes; those gauges are the instantaneous rates.
    let tokens_per_sec = parsed.gauges.get(PREDICTED_TOKENS_SECONDS).copied();
    let prompt_tokens_per_sec = parsed.gauges.get(PROMPT_TOKENS_SECONDS).copied();

    // Running average of positive readings, same pattern as the vLLM
    // adapter: stays stable while the engine idles.
    let push_avg = |accum: &mut (f64, u64), sample: Option<f64>| {
        if let Some(value) = sample {
            if value > 0.0 {
                accum.0 += value;
                accum.1 += 1;
            }
        }
        if accum.1 > 0 {
            Some(accum.0 / accum.1 as f64)
        } else {
            None
        }
    };
    let avg_tokens_per_sec = push_avg(avg_generation, tokens_per_sec);
    let avg_prompt_tokens_per_sec = push_avg(avg_prompt, prompt_tokens_per_sec);

    // Lifetime decode time per generated token: seconds / tokens. It is a
    // ratio over the engine's whole life, so there is no warmup skew to
    // screen. Serves both the TPOT tile and the inter-token gap tile —
    // llama.cpp reports one decode timing, and that is what both mean here.
    let (tpot_ms, per_request_tps) = match (generation_seconds, generation_total) {
        (Some(seconds), Some(tokens)) if seconds > 0.0 && tokens > 0.0 => {
            (Some(seconds / tokens * 1000.0), Some(tokens / seconds))
        }
        _ => (None, None),
    };
    let per_request_prompt_tps = match (prompt_seconds, prompt_total) {
        (Some(seconds), Some(tokens)) if seconds > 0.0 && tokens > 0.0 => Some(tokens / seconds),
        _ => None,
    };

    // `prompt_tokens_total` counts the tokens actually processed; the cached
    // ones are counted separately, so the tokens *queried* are their sum —
    // and the hit rate is the cached share of that sum. Only current
    // releases expose the cached counter; without it both fields stay blank.
    let (prefix_cache_hit_rate, prefix_cache_queries_total) = match (prompt_total, prompt_cached) {
        (Some(queried_new), Some(cached)) => {
            let queried = queried_new + cached;
            (
                (queried > 0.0).then(|| cached / queried * 100.0),
                Some(queried.max(0.0) as u64),
            )
        }
        _ => (None, None),
    };

    // Speculative decoding: same counter triple as vLLM, same derivations.
    let spec_draft = parsed.counters.get(SPEC_DECODE_DRAFT_TOKENS_TOTAL).copied();
    let spec_accepted = parsed
        .counters
        .get(SPEC_DECODE_ACCEPTED_TOKENS_TOTAL)
        .copied();
    let spec_drafts = parsed.counters.get(SPEC_DECODE_DRAFTS_TOTAL).copied();

    let spec_decode_acceptance_rate = spec_acceptance_rate(spec_accepted, spec_draft);
    let spec_decode_mean_acceptance_length =
        spec_mean_acceptance_length(spec_accepted, spec_drafts);
    // Live (windowed) TAR from the per-poll delta. A missing or backwards
    // counter (engine restart) discards the snapshot rather than diffing
    // against a stale value — identical rules to the vLLM adapter.
    let (spec_decode_acceptance_rate_live, next_prev_spec_decode) =
        match (spec_accepted, spec_draft) {
            (Some(accepted), Some(draft)) => {
                let live = prev_spec_decode.and_then(|(prev_acc, prev_draft)| {
                    if accepted >= prev_acc && draft >= prev_draft {
                        spec_acceptance_rate(Some(accepted - prev_acc), Some(draft - prev_draft))
                    } else {
                        None
                    }
                });
                (live, Some((accepted, draft)))
            }
            _ => (None, None),
        };

    // Accepted tokens per draft position: the labeled counter is indexed
    // by `position="N"`; gaps are zero-filled so the index stays positional.
    let spec_decode_accepted_tokens_per_pos = {
        let prefix = format!("{}{{", SPEC_DECODE_ACCEPTED_PER_POS_TOTAL);
        let mut items = parsed
            .labeled_counters
            .iter()
            .filter_map(|(key, value)| {
                let pos = key
                    .strip_prefix(&prefix)
                    .and_then(|s| s.strip_suffix('}'))
                    .and_then(|s| s.strip_prefix("position="))
                    .map(|s| s.trim_matches('"'))
                    .and_then(|s| s.parse::<usize>().ok())?;
                Some((pos, *value))
            })
            .collect::<Vec<_>>();
        if items.is_empty() {
            None
        } else {
            items.sort_by_key(|(p, _)| *p);
            let max = items.last().unwrap().0;
            let mut per_pos = vec![0u64; max + 1];
            for (p, v) in items {
                per_pos[p] = v.max(0.0) as u64;
            }
            Some(per_pos)
        }
    };

    let kv_cache_percent = parsed
        .gauges
        .get(KV_CACHE_USAGE_RATIO)
        .map(|&ratio| ratio.clamp(0.0, 1.0) * 100.0);

    let metrics = EngineMetrics {
        tokens_per_sec,
        avg_tokens_per_sec,
        per_request_tps,
        // llama.cpp reports no per-request TTFT.
        ttft_ms: None,
        active_requests: parsed
            .gauges
            .get(REQUESTS_PROCESSING)
            .map(|&v| v.max(0.0) as u64),
        queued_requests: parsed
            .gauges
            .get(REQUESTS_DEFERRED)
            .map(|&v| v.max(0.0) as u64),
        kv_cache_percent,
        // The ratio is the engine's own measurement, not an estimate.
        kv_cache_is_estimated: false,
        // llama.cpp exposes no total-request counter.
        total_requests: None,
        // No per-request histograms: end-to-end latency is not reported.
        e2e_latency_ms: None,
        prompt_tokens_per_sec,
        avg_prompt_tokens_per_sec,
        per_request_prompt_tps,
        swapped_requests: None,
        prefix_cache_hit_rate,
        queue_time_ms: None,
        inter_token_latency_ms: tpot_ms,
        preemptions_total: None,
        total_prompt_tokens: prompt_total.map(|v| v.max(0.0) as u64),
        total_generation_tokens: generation_total.map(|v| v.max(0.0) as u64),
        prefix_cache_queries_total,
        avg_batch_size: parsed.gauges.get(N_BUSY_SLOTS_PER_DECODE).copied(),
        ttft_percentiles: None,
        itl_percentiles: None,
        e2e_percentiles: None,
        ttft_goodput_pct: None,
        itl_goodput_pct: None,
        e2e_goodput_pct: None,
        ttft_buckets: None,
        itl_buckets: None,
        e2e_buckets: None,
        tpot_ms,
        tpot_percentiles: None,
        tpot_goodput_pct: None,
        tpot_buckets: None,
        spec_decode_draft_tokens_total: spec_draft.map(|v| v.max(0.0) as u64),
        spec_decode_accepted_tokens_total: spec_accepted.map(|v| v.max(0.0) as u64),
        spec_decode_drafts_total: spec_drafts.map(|v| v.max(0.0) as u64),
        spec_decode_acceptance_rate,
        spec_decode_acceptance_rate_live,
        spec_decode_mean_acceptance_length,
        spec_decode_accepted_tokens_per_pos,
        total_decode_calls: parsed
            .counters
            .get(N_DECODE_TOTAL)
            .map(|&v| v.max(0.0) as u64),
        max_sequence_tokens: parsed
            .counters
            .get(N_TOKENS_MAX)
            .map(|&v| v.max(0.0) as u64),
        warming_up: false,
    };

    (metrics, next_prev_spec_decode)
}

/// The `--model` value on the launch command line is usually a filesystem
/// path to a GGUF file; the display name wants the file itself. `hf:` URIs
/// (`hf:org/repo:quant`) carry the repo id — keep it whole.
fn display_model_hint(hint: &str) -> String {
    if let Some(hf_uri) = hint.strip_prefix("hf:") {
        return hf_uri.to_string();
    }
    if hint.contains('/') {
        hint.rsplit('/').next().unwrap_or(hint).to_string()
    } else {
        hint.to_string()
    }
}

pub struct LlamaAdapter {
    client: reqwest::Client,
    endpoint: String,
    /// Optional bearer token for `llama-server --api-key` deployments.
    /// Applied to engine requests; open endpoints ignore it harmlessly.
    api_key: Option<String>,
    /// `--model <path>` recovered from the launch command line during
    /// detection. Display fallback for when `/v1/models` cannot be read —
    /// the engine's own answer (an `--alias` or the GGUF file name) always
    /// wins when it is available.
    model_hint: Option<String>,
    /// Previous (accepted, draft) spec-decode counter readings for the live
    /// (windowed) token acceptance rate. No timestamp is stored because the
    /// live TAR is a unit-free ratio of deltas.
    prev_spec_decode: Mutex<Option<(f64, f64)>>,
    /// Running average for generation throughput: (sum of positive readings,
    /// count of readings).
    avg_accum: Mutex<(f64, u64)>,
    /// Running average for prompt throughput: (sum of positive readings,
    /// count of readings).
    avg_prompt_accum: Mutex<(f64, u64)>,
}

impl LlamaAdapter {
    pub fn new(
        client: reqwest::Client,
        endpoint: String,
        model_hint: Option<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            client,
            endpoint,
            api_key,
            model_hint,
            prev_spec_decode: Mutex::new(None),
            avg_accum: Mutex::new((0.0, 0)),
            avg_prompt_accum: Mutex::new((0.0, 0)),
        }
    }

    /// `/metrics` URL, optionally pinned to a model. Multi-model preset
    /// pools (`--models-preset`) reject a bare scrape with 400 "model name
    /// is missing from the request"; a single-model server accepts
    /// `?model=` harmlessly, so the retry is safe in either case.
    fn metrics_url(&self, model: Option<&str>) -> String {
        let bare = format!("{}/metrics", self.endpoint);
        let Ok(mut url) = reqwest::Url::parse(&bare) else {
            return bare;
        };
        if let Some(model) = model {
            url.query_pairs_mut().append_pair("model", model);
        }
        url.to_string()
    }

    fn auth(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => rb.bearer_auth(key),
            None => rb,
        }
    }

    /// What `/v1/models` had to say, reduced to what the name resolution
    /// needs. llama-server answers with an OpenAI-shaped list whose single
    /// id is the `--alias` or the GGUF file name.
    async fn models_reply(&self) -> Result<String, ModelMetadataError> {
        let resp = self
            .auth(
                self.client
                    .get(format!("{}/v1/models", self.endpoint))
                    .timeout(Duration::from_secs(2)),
            )
            .send()
            .await
            .map_err(|e| {
                tracing::debug!(endpoint = %self.endpoint, error = %e, "/v1/models request failed");
                ModelMetadataError::Unavailable
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = classify_models_error_status(status.as_u16());
            if error == ModelMetadataError::AuthRequired {
                tracing::warn!(
                    endpoint = %self.endpoint,
                    status = %status,
                    "/v1/models rejected the request as unauthorized — \
                     configure a provider API key to read model metadata",
                );
            } else {
                tracing::debug!(
                    endpoint = %self.endpoint,
                    status = %status,
                    "/v1/models returned non-success",
                );
            }
            return Err(error);
        }

        let id = resp
            .json::<OpenAIModelsResponse>()
            .await
            .ok()
            .and_then(|models| {
                models
                    .data
                    .first()
                    .map(|m| m.id.clone())
                    .filter(|id| !id.is_empty())
            });
        match id {
            Some(id) => Ok(id),
            None => {
                tracing::debug!(
                    endpoint = %self.endpoint,
                    "/v1/models listed no model id",
                );
                Err(ModelMetadataError::Unavailable)
            }
        }
    }
}

#[async_trait]
impl EngineAdapter for LlamaAdapter {
    fn engine_type(&self) -> EngineType {
        EngineType::Llama
    }

    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    async fn health_check(&self) -> EngineStatus {
        // /health is documented public, but the bearer is applied anyway so
        // the code matches the field's contract (harmless on open routes).
        match self
            .auth(
                self.client
                    .get(format!("{}/health", self.endpoint))
                    .timeout(Duration::from_secs(2)),
            )
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => EngineStatus::Running,
            Ok(r) => EngineStatus::Error(format!("HTTP {}", r.status())),
            Err(e) => EngineStatus::Error(e.to_string()),
        }
    }

    async fn get_model_info(&self) -> ModelResolution {
        // The engine's own id (an `--alias` or the GGUF file name) is the
        // display name when it exists — unlike vLLM, where a bare slug means
        // the router stripped the provider prefix, a bare llama.cpp id is
        // the canonical name.
        match self.models_reply().await {
            Ok(id) => ModelResolution {
                model: Some(ModelInfo {
                    name: normalize_model_id(&id),
                    parameter_size: None,
                    quantization: None,
                    precision: None,
                    tensor_type: None,
                    model_type: None,
                    pipeline_tag: None,
                }),
                metadata_error: None,
            },
            Err(metadata_error) => {
                let name = self
                    .model_hint
                    .as_deref()
                    .map(display_model_hint)
                    .map(|hint| normalize_model_id(&hint));
                ModelResolution {
                    model: name.map(|name| ModelInfo {
                        name,
                        parameter_size: None,
                        quantization: None,
                        precision: None,
                        tensor_type: None,
                        model_type: None,
                        pipeline_tag: None,
                    }),
                    metadata_error: Some(metadata_error),
                }
            }
        }
    }

    async fn get_metrics(&self) -> Option<EngineMetrics> {
        // A server started without `--metrics` answers /metrics with an HTTP
        // error and a JSON body (llama.cpp's `ERROR_TYPE_NOT_SUPPORTED`) —
        // not exposition text. The status is the reliable signal, so it is
        // checked before the body is read: the engine stays listed but
        // simply reports no metrics.
        let res = self
            .auth(
                self.client
                    .get(self.metrics_url(None))
                    .timeout(Duration::from_secs(5)),
            )
            .send()
            .await
            .ok()?;
        let res = if res.status().is_success() {
            res
        } else {
            // A multi-model preset pool needs the model name on the scrape.
            // Its own `/v1/models` id wins over the command-line hint.
            let model = self
                .models_reply()
                .await
                .ok()
                .or_else(|| self.model_hint.clone());
            let retry = match model {
                Some(model) => self
                    .auth(
                        self.client
                            .get(self.metrics_url(Some(&model)))
                            .timeout(Duration::from_secs(5)),
                    )
                    .send()
                    .await
                    .ok()?,
                None => return None,
            };
            if retry.status().is_success() {
                retry
            } else {
                tracing::debug!(
                    endpoint = %self.endpoint,
                    status = %retry.status(),
                    "/metrics unavailable (server started without `--metrics`?)",
                );
                return None;
            }
        };
        let body = res.text().await.ok()?;

        let parsed = parse_prometheus_text(&body)?;

        let prev_spec_decode = self.prev_spec_decode.lock().await.take();
        let (metrics, next_prev) = {
            let mut avg_accum = self.avg_accum.lock().await;
            let mut avg_prompt_accum = self.avg_prompt_accum.lock().await;
            map_metrics(
                &parsed,
                prev_spec_decode,
                &mut avg_accum,
                &mut avg_prompt_accum,
            )
        };
        *self.prev_spec_decode.lock().await = next_prev;

        Some(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A current-release (master) metrics body: the cumulative counters, the
    /// engine's own windowed throughput gauges, the request/slot gauges, the
    /// prompt-cache counter, and the speculative-decoding triple.
    const FULL_BODY: &str = "\
# HELP llamacpp:prompt_tokens_total Total prompt processing tokens (excluding cached tokens).
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 12345
# HELP llamacpp:prompt_tokens_cached_total Total prompt processing tokens (cached tokens).
# TYPE llamacpp:prompt_tokens_cached_total counter
llamacpp:prompt_tokens_cached_total 678
# HELP llamacpp:prompt_seconds_total Total time spent on prompt processing.
# TYPE llamacpp:prompt_seconds_total counter
llamacpp:prompt_seconds_total 12.345
# HELP llamacpp:tokens_predicted_total Total predicted tokens.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 987
# HELP llamacpp:tokens_predicted_seconds_total Total time spent on token generation.
# TYPE llamacpp:tokens_predicted_seconds_total counter
llamacpp:tokens_predicted_seconds_total 9.87
# HELP llamacpp:prompt_tokens_seconds Average prompt throughput between two scrapes.
# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds 50
# HELP llamacpp:predicted_tokens_seconds Average generation throughput between two scrapes.
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 25
# HELP llamacpp:requests_processing Total active requests.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 3
# HELP llamacpp:requests_deferred Total deferred requests.
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 1
# HELP llamacpp:n_busy_slots_per_decode Average number of busy slots per decode.
# TYPE llamacpp:n_busy_slots_per_decode gauge
llamacpp:n_busy_slots_per_decode 2
# HELP llamacpp:spec_decode_num_draft_tokens_total Total speculative draft tokens generated.
# TYPE llamacpp:spec_decode_num_draft_tokens_total counter
llamacpp:spec_decode_num_draft_tokens_total 1000
# HELP llamacpp:spec_decode_num_accepted_tokens_total Total speculative draft tokens accepted.
# TYPE llamacpp:spec_decode_num_accepted_tokens_total counter
llamacpp:spec_decode_num_accepted_tokens_total 800
# HELP llamacpp:spec_decode_num_drafts_total Total speculative decoding drafts.
# TYPE llamacpp:spec_decode_num_drafts_total counter
llamacpp:spec_decode_num_drafts_total 200
# HELP llamacpp:n_decode_total Total number of llama_decode() calls.
# TYPE llamacpp:n_decode_total counter
llamacpp:n_decode_total 543
# HELP llamacpp:n_tokens_max Largest observed sequence length (prompt + generation).
# TYPE llamacpp:n_tokens_max counter
llamacpp:n_tokens_max 2048
# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per draft position.
# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"} 800
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"} 400
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\"2\"} 120
";

    fn parse(body: &str) -> ParsedMetrics {
        parse_prometheus_text(body).expect("parse")
    }

    #[test]
    fn full_body_maps_every_field_llama_reports() {
        let (m, next) = map_metrics(&parse(FULL_BODY), None, &mut (0.0, 0), &mut (0.0, 0));

        // Raw lifetime counters.
        assert_eq!(m.total_prompt_tokens, Some(12345));
        assert_eq!(m.total_generation_tokens, Some(987));

        // Engine-provided windowed throughput, plus the running average of
        // the first positive reading.
        assert_eq!(m.tokens_per_sec, Some(25.0));
        assert_eq!(m.prompt_tokens_per_sec, Some(50.0));
        assert_eq!(m.avg_tokens_per_sec, Some(25.0));
        assert_eq!(m.avg_prompt_tokens_per_sec, Some(50.0));

        // Request/slot state.
        assert_eq!(m.active_requests, Some(3));
        assert_eq!(m.queued_requests, Some(1));
        assert_eq!(m.avg_batch_size, Some(2.0));

        // Lifetime decode-time derivations: 9.87s / 987 tokens = 10 ms/token,
        // 100 tokens/s; prefill 12345 / 12.345s = 1000 tokens/s.
        let close = |a: Option<f64>, b: f64| a.is_some_and(|v| (v - b).abs() < 1e-9);
        assert!(close(m.tpot_ms, 10.0));
        assert!(close(m.inter_token_latency_ms, 10.0));
        assert!(close(m.per_request_tps, 100.0));
        assert!(close(m.per_request_prompt_tps, 1000.0));

        // Prompt cache: 678 cached of 12345 + 678 queried.
        assert_eq!(m.prefix_cache_queries_total, Some(13023));
        assert!(
            (m.prefix_cache_hit_rate.unwrap() - 678.0 / 13023.0 * 100.0).abs() < 1e-9,
            "hit rate = 678/13023*100, got {:?}",
            m.prefix_cache_hit_rate
        );

        // Speculative decoding: lifetime 800/1000 = 80%, mean length 4.0.
        assert_eq!(m.spec_decode_draft_tokens_total, Some(1000));
        assert_eq!(m.spec_decode_accepted_tokens_total, Some(800));
        assert_eq!(m.spec_decode_drafts_total, Some(200));
        assert_eq!(m.spec_decode_acceptance_rate, Some(80.0));
        assert_eq!(
            m.spec_decode_acceptance_rate_live, None,
            "no previous reading yet"
        );
        assert_eq!(m.spec_decode_mean_acceptance_length, Some(4.0));
        assert_eq!(next, Some((800.0, 1000.0)));

        // Decode-call counter, largest observed sequence length, and the
        // per-draft-position accepted-token distribution (indexed by
        // position).
        assert_eq!(m.total_decode_calls, Some(543));
        assert_eq!(m.max_sequence_tokens, Some(2048));
        assert_eq!(
            m.spec_decode_accepted_tokens_per_pos.as_deref(),
            Some([800, 400, 120].as_slice())
        );

        // No warmup machinery applies — llama.cpp has no per-request
        // histograms to screen.
        assert!(!m.warming_up);
    }

    #[test]
    fn fields_without_a_llama_cpp_equivalent_stay_blank() {
        let (m, _) = map_metrics(&parse(FULL_BODY), None, &mut (0.0, 0), &mut (0.0, 0));

        assert_eq!(m.ttft_ms, None);
        assert_eq!(m.total_requests, None);
        assert_eq!(m.e2e_latency_ms, None);
        // The percentile/bucket types carry no PartialEq, so absence is
        // asserted on the Option itself.
        assert!(m.e2e_percentiles.is_none());
        assert!(m.e2e_buckets.is_none());
        assert!(m.ttft_percentiles.is_none());
        assert!(m.ttft_buckets.is_none());
        assert!(m.itl_percentiles.is_none());
        assert!(m.itl_buckets.is_none());
        assert!(m.tpot_percentiles.is_none());
        assert!(m.tpot_buckets.is_none());
        assert_eq!(m.ttft_goodput_pct, None);
        assert_eq!(m.itl_goodput_pct, None);
        assert_eq!(m.e2e_goodput_pct, None);
        assert_eq!(m.tpot_goodput_pct, None);
        assert_eq!(
            m.kv_cache_percent, None,
            "master no longer reports the usage ratio"
        );
        assert!(!m.kv_cache_is_estimated);
        assert_eq!(m.swapped_requests, None);
        assert_eq!(m.queue_time_ms, None);
        assert_eq!(m.preemptions_total, None);
    }

    #[test]
    fn kv_cache_usage_ratio_maps_to_percent_when_present() {
        // Older releases expose the ratio as a gauge in [0, 1].
        let (m, _) = map_metrics(
            &parse(
                "
# HELP llamacpp:kv_cache_usage_ratio KV cache usage ratio.
# TYPE llamacpp:kv_cache_usage_ratio gauge
llamacpp:kv_cache_usage_ratio 0.42
",
            ),
            None,
            &mut (0.0, 0),
            &mut (0.0, 0),
        );
        assert_eq!(m.kv_cache_percent, Some(42.0));
        assert!(
            !m.kv_cache_is_estimated,
            "the engine-reported ratio is a measurement, not an estimate"
        );
    }

    #[test]
    fn prefix_cache_fields_stay_blank_without_the_cached_counter() {
        // Releases before the cached-token counter existed.
        let (m, _) = map_metrics(
            &parse(
                "
# HELP llamacpp:prompt_tokens_total Total prompt processing tokens.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 100
",
            ),
            None,
            &mut (0.0, 0),
            &mut (0.0, 0),
        );
        assert_eq!(m.prefix_cache_hit_rate, None);
        assert_eq!(m.prefix_cache_queries_total, None);
    }

    #[test]
    fn per_position_acceptance_stays_blank_without_the_labeled_counter() {
        // Releases without the per-position counter, and scrapes where
        // speculative decoding never ran, must leave the vector absent.
        let (m, _) = map_metrics(
            &parse(
                "
# HELP llamacpp:tokens_predicted_total Total predicted tokens.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 42
",
            ),
            None,
            &mut (0.0, 0),
            &mut (0.0, 0),
        );
        assert!(m.spec_decode_accepted_tokens_per_pos.is_none());
        assert_eq!(m.total_decode_calls, None);
        assert_eq!(m.max_sequence_tokens, None);
    }

    #[test]
    fn decode_time_derivation_guards_zero_and_missing_denominators() {
        let body = "
# HELP llamacpp:tokens_predicted_seconds_total Total time spent on token generation.
# TYPE llamacpp:tokens_predicted_seconds_total counter
llamacpp:tokens_predicted_seconds_total 0
# HELP llamacpp:tokens_predicted_total Total predicted tokens.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 0
";
        let (m, _) = map_metrics(&parse(body), None, &mut (0.0, 0), &mut (0.0, 0));
        assert_eq!(m.tpot_ms, None);
        assert_eq!(m.inter_token_latency_ms, None);
        assert_eq!(m.per_request_tps, None);
        assert_eq!(m.total_generation_tokens, Some(0));
    }

    #[test]
    fn live_spec_decode_rate_comes_from_the_per_poll_delta() {
        let mut parsed = parse(FULL_BODY);
        // The second scrape: 1100 drafted, 850 accepted so far.
        parsed
            .counters
            .insert(SPEC_DECODE_DRAFT_TOKENS_TOTAL.to_string(), 1100.0);
        parsed
            .counters
            .insert(SPEC_DECODE_ACCEPTED_TOKENS_TOTAL.to_string(), 850.0);

        let (m, next) = map_metrics(&parsed, Some((800.0, 1000.0)), &mut (0.0, 0), &mut (0.0, 0));
        // Δ850-800 / Δ1100-1000 = 50%.
        assert_eq!(m.spec_decode_acceptance_rate_live, Some(50.0));
        assert_eq!(next, Some((850.0, 1100.0)));
    }

    #[test]
    fn backwards_spec_decode_counters_discard_the_snapshot() {
        // An engine restart resets counters; diffing against the stale
        // snapshot would negate the rate.
        let mut parsed = parse(FULL_BODY);
        parsed
            .counters
            .insert(SPEC_DECODE_DRAFT_TOKENS_TOTAL.to_string(), 50.0);
        parsed
            .counters
            .insert(SPEC_DECODE_ACCEPTED_TOKENS_TOTAL.to_string(), 40.0);

        let (m, next) = map_metrics(&parsed, Some((800.0, 1000.0)), &mut (0.0, 0), &mut (0.0, 0));
        assert_eq!(m.spec_decode_acceptance_rate_live, None);
        assert_eq!(next, Some((40.0, 50.0)), "fresh snapshot recorded");
    }

    #[test]
    fn avg_accumulator_keeps_running_average_across_scrapes() {
        // First scrape: 25 tok/s. Second: idle (0), third: 75 tok/s.
        let mut gen_accum = (0.0, 0u64);
        let mut prompt_accum = (0.0, 0u64);

        let m1 = map_metrics(&parse(FULL_BODY), None, &mut gen_accum, &mut prompt_accum);
        assert_eq!(m1.0.avg_tokens_per_sec, Some(25.0));

        let mut idle = parse(FULL_BODY);
        idle.gauges
            .insert(PREDICTED_TOKENS_SECONDS.to_string(), 0.0);
        idle.gauges.insert(PROMPT_TOKENS_SECONDS.to_string(), 0.0);
        let m2 = map_metrics(
            &idle,
            Some((800.0, 1000.0)),
            &mut gen_accum,
            &mut prompt_accum,
        );
        assert_eq!(m2.0.tokens_per_sec, Some(0.0));
        assert_eq!(
            m2.0.avg_tokens_per_sec,
            Some(25.0),
            "idle reading does not accumulate"
        );

        let mut busy = parse(FULL_BODY);
        busy.gauges
            .insert(PREDICTED_TOKENS_SECONDS.to_string(), 75.0);
        let m3 = map_metrics(
            &busy,
            Some((850.0, 1100.0)),
            &mut gen_accum,
            &mut prompt_accum,
        );
        assert_eq!(m3.0.avg_tokens_per_sec, Some(50.0), "(25 + 75) / 2");
    }

    #[test]
    fn empty_body_maps_to_all_blank_metrics() {
        let (m, next) = map_metrics(&parse(""), None, &mut (0.0, 0), &mut (0.0, 0));
        assert_eq!(m.tokens_per_sec, None);
        assert_eq!(m.total_prompt_tokens, None);
        assert_eq!(m.active_requests, None);
        assert_eq!(next, None);
    }

    #[test]
    fn display_hint_shows_the_file_not_the_path() {
        assert_eq!(
            display_model_hint("/mnt/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"),
            "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
        );
        // hf: URIs carry the repo id — keep it, drop the scheme marker.
        assert_eq!(
            display_model_hint("hf:Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"),
            "Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"
        );
        // Plain slugs and aliases pass through.
        assert_eq!(display_model_hint("my-alias"), "my-alias");
    }

    #[test]
    fn metrics_url_pins_the_model_for_preset_pools() {
        let adapter = LlamaAdapter::new(
            reqwest::Client::new(),
            "http://localhost:9931".into(),
            None,
            None,
        );
        assert_eq!(adapter.metrics_url(None), "http://localhost:9931/metrics");
        assert_eq!(
            adapter.metrics_url(Some("Qwen3.8-27B-UD-Q4_K_M")),
            "http://localhost:9931/metrics?model=Qwen3.8-27B-UD-Q4_K_M",
        );
        // Alias names with spaces must arrive URL-encoded (+ decodes to a
        // space server-side, like any standard query form encoding).
        assert_eq!(
            adapter.metrics_url(Some("my model v2")),
            "http://localhost:9931/metrics?model=my+model+v2",
        );
    }

    #[test]
    fn create_adapter_builds_the_llama_adapter() {
        let client = reqwest::Client::new();
        let adapter = super::super::create_adapter(
            EngineType::Llama,
            "http://localhost:8080".into(),
            client,
            Some("/models/m.gguf".into()),
            None,
        );
        assert_eq!(adapter.engine_type(), EngineType::Llama);
        assert_eq!(adapter.endpoint(), "http://localhost:8080");
    }
}
