"""Methods 3/4: decide BEFORE prediction, update EVERY original grid interval.

Unlike method 2 (grid_skip), a reused node never omits its latent update. The only difference between 3 and 4 is how a reused prediction is made.
Cache tensors belong to the trusted sampler, not mutable Agent policy memory.
"""
import time

from .context import Context, Prediction, validate_skip
from .models.cache import prepare_feature, refresh_prediction, reuse_prediction, reuse_residual
from .programs import digest
from .recovery import (FailureLog, POLICY_ERRORS, PolicyRecovery, clone_state, fatal_error,
                       policy_deadline, require_finite, restore_rng, rng_state, save_crash)
from .wrapper import _analyze


def sample_prediction_cache(*args, **kwargs):
    """3. Reuse the last final prediction; no current denoiser/output-head call."""
    return _sample_cache(*args, method="prediction_cache", **kwargs)


def sample_residual_cache(*args, **kwargs):
    """4. Reuse the whole backbone delta, then execute the current output head."""
    return _sample_cache(*args, method="residual_cache", **kwargs)


def _analyze_cache(programs, ctx, memory, timeout):
    if getattr(programs.analysis, "FEATURE_LAYERS", []):
        raise ValueError("Cache methods expose input_feature before prediction; deep FEATURE_LAYERS are unavailable")
    return _analyze(programs, ctx, memory, timeout)


def _sample_cache(adapter, programs, prompt, seed, reference_steps, max_nfe, schedule, recorder,
                  agent=None, policy_timeout_seconds=30, method="prediction_cache", max_consecutive_skips=4):
    if type(max_consecutive_skips) is not int or max_consecutive_skips < 0:
        raise ValueError("max_consecutive_skips must be a nonnegative integer")
    if reference_steps < 1 or max_nfe < 1 or policy_timeout_seconds <= 0:
        raise ValueError("Steps, prediction budget and policy timeout must be positive")
    if max_nfe < (reference_steps + max_consecutive_skips) // (max_consecutive_skips + 1):
        raise ValueError("Prediction budget cannot satisfy the maximum consecutive reuse limit")
    started = time.perf_counter()
    failures = FailureLog(agent.root if agent else recorder.run_root)
    first_failure = len(failures.records)
    recovery = PolicyRecovery(programs)
    # All caches reset here on EVERY image, including evaluation without an Agent.
    cached_prediction = cached_residual = reference_sigma = None
    previous_latent = previous_feature = previous_sigma = previous_step = None
    memory, history, kept, reused, timesteps, sigma_schedule = {}, [], [], [], [], []
    consecutive_skips = recovered_steps = default_steps = 0
    feature_seconds = policy_seconds = denoiser_seconds = reuse_seconds = 0.0
    agent_before = agent.seconds if agent else 0.0
    calls_before = agent.count if agent else 0
    ctx, stage = None, "start"
    try:
        latent, sigmas = adapter.start(prompt, seed, reference_steps)
        total = len(sigmas) - 1
        sampling_started = time.perf_counter()
        sigma_schedule.append(sigmas[0])
        for step, (sigma, next_sigma) in enumerate(zip(sigmas, sigmas[1:])):
            stage = "cache_feature"
            require_finite(latent, "input latent")
            adapter.synchronize()
            began = time.perf_counter()
            feature = prepare_feature(adapter, latent, sigma)
            require_finite(feature, "input feature")
            adapter.synchronize()
            feature_seconds += time.perf_counter() - began
            ctx = Context(prompt, seed, step, adapter.denoiser_evaluations, max_nfe, sigma, sigmas[-1],
                          adapter.timestep(sigma), tuple(sigmas), latent, Prediction(None), history,
                          sample_id=recorder.root.name, method=method,
                          input_feature=feature, reference_feature=previous_feature,
                          reference_latent=previous_latent, observation_reference_step=previous_step,
                          observation_reference_sigma=previous_sigma,
                          cache_observation={
                              "sigma": sigma, "step_delta_sigma": sigma - next_sigma,
                              "since_real_delta_sigma": reference_sigma - sigma if reference_sigma is not None else 0.0,
                              "consecutive_skips": consecutive_skips, "has_reference": cached_prediction is not None})
            step_rng = rng_state()
            cursor = len(failures.records)
            changed = False
            decision_status = None
            old_version = programs.version
            # Agent calls and repeat inspect see Prediction(None): no current-output leakage.
            while True:
                stage = "analysis"
                began = time.perf_counter()
                try:
                    ctx.analysis_memory = clone_state(memory)
                    analyzed = _analyze_cache(programs, ctx, memory, policy_timeout_seconds)
                    break
                except POLICY_ERRORS as exc:
                    if fatal_error(exc) or programs.version == getattr(programs, "default_hash", None):
                        raise
                    bad = programs.version
                    recovery.fallback()
                    memory = {}
                    ctx.analysis_memory = {}
                    restore_rng(step_rng)
                    failures.record(exc, stage=stage, ctx=ctx, version=bad, outcome="reverted",
                                    restored=True, target=programs.version, memory="reset")
                    if agent:
                        agent.mark_reverted(bad, failures.records[-1])
                finally:
                    policy_seconds += time.perf_counter() - began
            if agent and schedule.due(step):
                agent.failure_log = failures
                old_version = programs.version
                stage = "agent"
                try:
                    answer = agent.check(ctx, analyzed[0], programs)
                    stage = "load_candidate"
                    source = answer.get("candidate_workspace") if isinstance(answer, dict) else None
                    version = digest(source or programs.workspace)
                    if source and answer.get("candidate_version") != version:
                        raise ValueError("Candidate changed after the Agent response was validated")
                    if version in programs.recovery_bad:
                        raise ValueError("Candidate version has already failed")
                    changed = programs.reload(source, timeout_seconds=policy_timeout_seconds)
                    decision_status = "applied" if changed else "kept"
                    analyzed = None if changed else analyzed
                except POLICY_ERRORS as exc:
                    if fatal_error(exc):
                        raise
                    bad = digest(programs.workspace)
                    if stage == "load_candidate" and bad != old_version:
                        programs.recovery_bad.add(bad)
                    programs.restore(old_version)
                    failures.record(exc, stage=stage, ctx=ctx, version=bad, restored=True, target=old_version)
                    decision_status, analyzed = "rejected", None
                finally:
                    restore_rng(step_rng)
            while True:
                began = time.perf_counter()
                try:
                    stage = "analysis"
                    if analyzed is None:
                        analyzed = _analyze_cache(programs, ctx, memory, policy_timeout_seconds)
                    observation, proposed_memory, working = analyzed
                    stage = "updater"
                    with policy_deadline(policy_timeout_seconds):
                        action = programs.updater.choose(clone_state(observation), working, proposed_memory)
                    skip, forced = validate_skip(action), None
                    if cached_prediction is None:
                        skip, forced = False, "initialization"
                    elif consecutive_skips >= max_consecutive_skips:
                        skip, forced = False, "reuse_limit"
                    elif ctx.nfe + 1 + (total - step - 1) // (max_consecutive_skips + 1) > max_nfe:
                        # Reserve enough real predictions to honor future mandatory refreshes.
                        skip, forced = True, "budget"
                    if not skip and ctx.nfe >= max_nfe:
                        raise ValueError("Prediction budget exhausted before a mandatory cache refresh")
                    break
                except POLICY_ERRORS as exc:
                    if fatal_error(exc) or programs.version == getattr(programs, "default_hash", None):
                        raise
                    bad = programs.version
                    if changed:
                        programs.recovery_bad.add(bad)
                        programs.restore(old_version)
                        changed, decision_status = False, "rejected"
                    else:
                        recovery.fallback()
                    memory, analyzed = {}, None
                    ctx.analysis_memory = {}
                    restore_rng(step_rng)
                    failures.record(exc, stage=stage, ctx=ctx, version=bad, outcome="reverted",
                                    restored=True, target=programs.version, memory="reset")
                    if agent:
                        agent.mark_reverted(bad, failures.records[-1])
                finally:
                    policy_seconds += time.perf_counter() - began
            # Policies have now finished. Only trusted code can touch model/cache/solver state.
            stage = "cache_reuse" if skip else "prediction"
            adapter.synchronize()
            began = time.perf_counter()
            residual = cached_residual
            if not skip:
                prediction, residual = refresh_prediction(adapter, latent, sigma, feature,
                                                         residual=method == "residual_cache")
            elif method == "prediction_cache":
                prediction = reuse_prediction(cached_prediction)
            else:
                prediction = reuse_residual(adapter, latent, sigma, feature, cached_residual)
            adapter.synchronize()
            elapsed = time.perf_counter() - began
            if skip:
                reuse_seconds += elapsed
            else:
                denoiser_seconds += elapsed
            require_finite(prediction.guided, "model prediction")
            if residual is not None:
                require_finite(residual, "backbone residual")
            stage = "scheduler"
            next_latent = adapter.advance(clone_state(latent), clone_state(prediction), sigma, next_sigma)
            require_finite(next_latent, "updated latent")
            for attribute in ("shape", "dtype", "device"):
                if getattr(next_latent, attribute, None) != getattr(latent, attribute, None):
                    raise ValueError(f"Updated latent changed {attribute}")
            # Commit caches only after the numerical update succeeds.
            if skip:
                reused.append(step)
                consecutive_skips += 1
            else:
                kept.append(step)
                cached_prediction, cached_residual = clone_state(prediction), residual
                reference_sigma = sigma
                consecutive_skips = 0
            if changed:
                programs.restore_workspace()
            memory = proposed_memory
            recovery.success()
            failures.recovered(failures.records[cursor:], programs.version)
            recovered_steps += len(failures.records) > cursor
            default_steps += programs.version == getattr(programs, "default_hash", None)
            if decision_status is not None:
                agent.record_outcome(decision_status, ctx, programs.version,
                                     [entry["id"] for entry in failures.records[cursor:]])
                recorder.event("decision", step=step, nfe=ctx.nfe, version=programs.version, changed=changed)
            record = {"step": step, "nfe": adapter.denoiser_evaluations, "sigma": sigma,
                      "next_sigma": next_sigma, "timestep": ctx.timestep,
                      "next_timestep": adapter.timestep(next_sigma), "version": programs.version,
                      "method": method, "observation": observation, "reuse_cache": skip, "requested_skip": action.skip,
                      "forced": forced, "reference_sigma": reference_sigma,
                      "backbone_evaluated": not skip, "scheduler_updated": True, "output_head_evaluated": not skip or method == "residual_cache"}
            if agent:
                agent.record_execution(ctx, record)
            recorder.event("step", **record)
            history.append({**record, "latent": latent, "prediction": clone_state(prediction)})
            if agent is None:
                limit = getattr(programs.analysis, "HISTORY_LIMIT", 0)
                if limit is not None:
                    del history[:max(0, len(history) - limit)]
            timesteps.append(ctx.timestep)
            sigma_schedule.append(next_sigma)
            previous_latent, previous_feature = latent, feature
            previous_sigma, previous_step = sigma, step
            latent = next_latent
        adapter.synchronize()
        sampling_seconds = time.perf_counter() - sampling_started
        stage = "finish"
        image = adapter.finish(latent)
        adapter.synchronize()
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        failures.record(exc, stage=stage, ctx=ctx, version=programs.version,
                        outcome="interrupted" if isinstance(exc, KeyboardInterrupt) else "aborted")
        save_crash(adapter, ctx, recorder, programs, exc)
        recorder.event("sampling_aborted", stage=stage, error=str(exc), version=programs.version)
        raise
    failure_count = len(failures.records) - first_failure
    result = {"method": method, "nfe": adapter.denoiser_evaluations, "steps": total,
              "denoiser_items": adapter.denoiser_items, "timesteps": timesteps,
              "sigma_schedule": sigma_schedule, "reference_sigmas": list(sigmas),
              "kept_indices": kept, "reused_indices": reused, "skipped_indices": [],
              "scheduler_updates": total, "feature_extractions": total,
              "output_head_evaluations": total if method == "residual_cache" else len(kept),
              "max_consecutive_skips": max_consecutive_skips,
              "feature_seconds": feature_seconds, "policy_seconds": policy_seconds,
              "denoiser_seconds": denoiser_seconds, "cache_reuse_seconds": reuse_seconds,
              "sampling_seconds": sampling_seconds, "end_to_end_seconds": time.perf_counter() - started,
              "agent_seconds": agent.seconds - agent_before if agent else 0.0,
              "agent_calls": agent.count - calls_before if agent else 0,
              "final_policy_version": programs.version, "terminal_sigma": sigmas[-1],
              "failure_count": failure_count, "recovered": bool(recovered_steps),
              "recovery_steps": recovered_steps, "default_fallback_steps": default_steps,
              "valid_for_benchmark": failure_count == 0, "retry_nfe": 0}
    return image, result, history
