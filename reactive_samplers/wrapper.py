"""Predict once, attempt a policy step, and commit only a valid numerical update."""
import json
import time
from dataclasses import replace

from .context import Action, Context, METHODS, Prediction, SkipAction, validate_action, validate_skip
from .programs import digest
from .recovery import (FailureLog, POLICY_ERRORS, PolicyRecovery, clone_state, fatal_error,
                       policy_deadline, require_finite, restore_rng, rng_state, save_crash)


def _analyze(programs, ctx, memory, timeout_seconds=30):
    working, proposed_memory = clone_state(ctx), clone_state(memory)
    with policy_deadline(timeout_seconds):
        observation = programs.analysis.analyze(working, proposed_memory)
    if not isinstance(observation, dict):
        raise TypeError("analysis.analyze must return a JSON observation dictionary")
    json.dumps(observation, allow_nan=False)
    limit = getattr(programs.analysis, "HISTORY_LIMIT", 2)
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("HISTORY_LIMIT must be a nonnegative integer or None")
    return observation, proposed_memory, working


def _features(adapter, programs):
    layers = getattr(programs.analysis, "FEATURE_LAYERS", [])
    if not isinstance(layers, (list, tuple)) or not all(isinstance(layer, str) for layer in layers):
        raise TypeError("FEATURE_LAYERS must be a list or tuple of names")
    if hasattr(adapter, "validate_feature_layers"):
        adapter.validate_feature_layers(layers)
    return layers


def _choose(programs, ctx, analyzed, adapter, timeout):
    decisions = []
    candidate = ctx
    while True:
        observation, proposed_memory, working = analyzed
        terminal = ctx.method == "grid_skip" and candidate.candidate_index == len(ctx.reference_sigmas) - 1
        if terminal:
            action = SkipAction(False)  # An integration boundary, not another denoiser evaluation.
        else:
            with policy_deadline(timeout):
                action = programs.updater.choose(clone_state(observation), working, proposed_memory)
        if ctx.method == "adaptive_sigma":
            return action, observation, proposed_memory, decisions
        skip = validate_skip(action, candidate)
        forced = "terminal" if terminal else "budget" if ctx.remaining_nfe == 0 else None
        if forced == "budget":
            skip = True
        decisions.append({"index": candidate.candidate_index, "sigma": candidate.candidate_sigma,
                          "timestep": candidate.candidate_timestep, "skip": skip, "forced": forced})
        if not skip:
            return Action(candidate.candidate_sigma), observation, proposed_memory, decisions
        index = candidate.candidate_index + 1
        candidate = replace(ctx, candidate_index=index,
                            candidate_timestep=adapter.timestep(ctx.reference_sigmas[index]))
        analyzed = _analyze(programs, candidate, proposed_memory, timeout)


def sample(adapter, programs, prompt, seed, reference_steps, max_nfe, schedule, recorder, agent=None,
           policy_timeout_seconds=30, method="adaptive_sigma"):
    if method not in METHODS or getattr(programs, "method", "adaptive_sigma") != method:
        raise ValueError("Sampling method and policy method must match")
    if max_nfe < 1 or reference_steps < 1:
        raise ValueError("reference_steps and max_nfe must be positive")
    if policy_timeout_seconds <= 0:
        raise ValueError("policy_timeout_seconds must be positive")
    started = time.perf_counter()
    failures = FailureLog(agent.root if agent else recorder.run_root)
    first_failure = len(failures.records)
    recovery = PolicyRecovery(programs)
    ctx, stage = None, "start"
    memory, history, timesteps, sigma_schedule = {}, [], [], []
    recovered_steps, default_steps = 0, 0
    denoiser_seconds = 0.0
    model_index, kept_indices, skipped_indices, grid_checks = 0, [], [], 0
    agent_before = agent.seconds if agent else 0.0
    decisions_before = agent.count if agent else 0
    try:
        latent, sigmas = adapter.start(prompt, seed, reference_steps)
        sigma, terminal = sigmas[0], sigmas[-1]
        sigma_schedule.append(sigma)
        sampling_started = time.perf_counter()
        for step in range(max_nfe):
            ctx = Context(prompt, seed, step, step, max_nfe, sigma, terminal,
                          adapter.timestep(sigma), tuple(sigmas), latent, Prediction(None), history,
                          sample_id=recorder.root.name)
            ctx.method = method
            if method == "grid_skip":
                ctx.model_index, ctx.candidate_index = model_index, model_index + 1
                ctx.candidate_timestep = adapter.timestep(sigmas[ctx.candidate_index])
            pending_failures = []
            step_failure_cursor = len(failures.records)
            step_rng = rng_state()

            def revert(exc, where):
                nonlocal memory
                if fatal_error(exc) or programs.version == getattr(programs, "default_hash", None):
                    raise exc
                bad = programs.version
                used_default = recovery.fallback()
                memory = {}  # Different code never receives the failed version's private state.
                ctx.analysis_memory = {}
                restore_rng(step_rng)
                for version, restore_error in recovery.restore_failures:
                    pending_failures.append(failures.record(
                        restore_error, stage="restore_version", ctx=ctx, version=version,
                        outcome="reverted", restored=True, target=programs.version, memory="reset"))
                entry = failures.record(exc, stage=where, ctx=ctx, version=bad,
                                        outcome="fallback_default" if used_default else "reverted",
                                        restored=True, target=programs.version, memory="reset")
                pending_failures.append(entry)
                recorder.event("policy_reverted", failure_id=entry["id"], step=step,
                               version=bad, target_version=programs.version)
                if agent:
                    agent.mark_reverted(bad, entry)

            while True:
                stage = "feature_layers"
                try:
                    layers = _features(adapter, programs)
                    break
                except POLICY_ERRORS as exc:
                    revert(exc, stage)
            stage = "prediction"
            require_finite(latent, "input latent")
            adapter.synchronize()
            prediction_started = time.perf_counter()
            prediction = adapter.predict(latent, sigma, layers)
            adapter.synchronize()
            denoiser_seconds += time.perf_counter() - prediction_started
            ctx.prediction, ctx.nfe = prediction, step + 1
            require_finite(prediction.guided, "model prediction")
            # The only solver state that can change during a policy attempt is snapshotted.
            solver_state = adapter.snapshot_state() if hasattr(adapter, "snapshot_state") else None
            while True:
                stage = "analysis"
                try:
                    ctx.analysis_memory = clone_state(memory)
                    analyzed = _analyze(programs, ctx, memory, policy_timeout_seconds)
                    break
                except POLICY_ERRORS as exc:
                    revert(exc, stage)

            pending_candidate, changed, decision_status = False, False, None
            previous_version = programs.version
            if agent and schedule.due(step):
                stage = "agent"
                candidate_version = previous_version
                agent.failure_log = failures
                try:
                    answer = agent.check(ctx, analyzed[0], programs)
                    stage = "load_candidate"
                    source = answer.get("candidate_workspace") if isinstance(answer, dict) else None
                    candidate_version = digest(source or programs.workspace)
                    if source and answer.get("candidate_version") != candidate_version:
                        raise ValueError("Candidate changed after the Agent response was validated")
                    if candidate_version in programs.recovery_bad:
                        raise ValueError("Candidate version has already failed; revise it before resubmitting")
                    changed = programs.reload(source, timeout_seconds=policy_timeout_seconds)
                    pending_candidate = changed
                    decision_status = "applied" if changed else "kept"
                    if changed:
                        analyzed = None  # Rerun the new probe at this same paused prediction.
                        restore_rng(step_rng)
                except POLICY_ERRORS as exc:
                    if fatal_error(exc):
                        raise
                    rejected = digest(programs.workspace) if stage == "agent" else candidate_version
                    if stage == "load_candidate" and rejected != previous_version:
                        programs.recovery_bad.add(rejected)
                    programs.restore(previous_version)
                    restore_rng(step_rng)
                    analyzed = None
                    entry = failures.record(exc, stage=stage, ctx=ctx, version=rejected,
                                            restored=True, target=previous_version,
                                            decision=agent.count)
                    pending_failures.append(entry)
                    recorder.event("decision_rejected", step=step, error=str(exc), version=previous_version,
                                   failure_id=entry["id"])
                    decision_status = "rejected"

            while True:
                try:
                    stage = "feature_layers"
                    _features(adapter, programs)  # Validate newly requested hooks before accepting a version.
                    stage = "analysis"
                    if analyzed is None:
                        analyzed = _analyze(programs, ctx, memory, policy_timeout_seconds)
                    stage = "updater"
                    action, observation, proposed_memory, decisions = _choose(
                        programs, ctx, analyzed, adapter, policy_timeout_seconds)
                    proposed_memory = clone_state(proposed_memory)
                    stage = "action"
                    next_sigma = validate_action(action, ctx)  # Always use trusted coordinates/budget.
                    stage = "scheduler"
                    next_latent = adapter.advance(clone_state(latent), clone_state(prediction), sigma, next_sigma)
                    require_finite(next_latent, "updated latent")
                    for attribute in ("shape", "dtype", "device"):
                        if getattr(next_latent, attribute, None) != getattr(latent, attribute, None):
                            raise ValueError(f"Updated latent changed {attribute}")
                    break
                except POLICY_ERRORS as exc:
                    if solver_state is not None:
                        adapter.restore_state(solver_state)
                    restore_rng(step_rng)
                    analyzed = None
                    if fatal_error(exc):
                        raise
                    if pending_candidate:
                        bad = programs.version
                        programs.recovery_bad.add(bad)
                        programs.restore(previous_version)
                        entry = failures.record(exc, stage=stage, ctx=ctx, version=bad, restored=True,
                                                target=previous_version, decision=agent.count)
                        pending_failures.append(entry)
                        recorder.event("decision_rejected", step=step, error=str(exc), version=previous_version,
                                       failure_id=entry["id"])
                        pending_candidate, decision_status = False, "rejected"
                    else:
                        revert(exc, stage)

            # All fallible policy/solver work succeeded. Only now publish state and trajectory.
            if pending_candidate:
                programs.restore_workspace()
            memory = proposed_memory
            recovery.success()
            failures.recovered(failures.records[step_failure_cursor:], programs.version)
            recovered_steps += bool(pending_failures)
            default_steps += programs.version == getattr(programs, "default_hash", None)
            if decision_status is not None:
                agent.record_outcome(decision_status, ctx, programs.version,
                                     [entry["id"] for entry in pending_failures])
                recorder.event("decision", step=step, nfe=ctx.nfe, version=programs.version,
                               changed=changed and decision_status == "applied")
            record = {"step": step, "nfe": ctx.nfe, "sigma": sigma, "next_sigma": next_sigma,
                      "timestep": ctx.timestep, "next_timestep": adapter.timestep(next_sigma),
                      "version": programs.version, "observation": observation}
            if method == "grid_skip":
                record.update(model_index=model_index, grid_decisions=decisions)
                if decision_status is not None:
                    record.update(agent_decision=agent.count, agent_outcome=decision_status,
                                  agent_output=str(agent.current_folder / "answer.txt"))
                kept_indices.append(model_index)
                skipped_indices.extend(item["index"] for item in decisions if item["skip"])
                grid_checks += len(decisions)
                model_index = decisions[-1]["index"]
            recorder.event("step", **record)
            timesteps.append(ctx.timestep)
            sigma_schedule.append(next_sigma)
            history.append({**record, "latent": latent,
                            "prediction": Prediction(prediction.guided, prediction.conditional,
                                                     prediction.unconditional)})
            if agent is None:
                limit = getattr(programs.analysis, "HISTORY_LIMIT", 2)
                if limit is not None:
                    del history[:max(0, len(history) - limit)]
            latent, sigma = next_latent, next_sigma
            if sigma == terminal:
                break
        stage = "finish"
        adapter.synchronize()
        sampling_seconds = time.perf_counter() - sampling_started
        image = adapter.finish(latent)
        adapter.synchronize()
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        if ctx is not None and stage == "prediction":
            ctx.nfe = getattr(adapter, "denoiser_evaluations", ctx.nfe + 1)
        failures.record(exc, stage=stage, ctx=ctx, version=programs.version,
                        outcome="interrupted" if isinstance(exc, KeyboardInterrupt) else "aborted")
        save_crash(adapter, ctx, recorder, programs, exc)
        recorder.event("sampling_aborted", stage=stage, error=str(exc), version=programs.version)
        raise
    failure_count = len(failures.records) - first_failure
    result = {"nfe": getattr(adapter, "denoiser_evaluations", step + 1), "steps": step + 1, "timesteps": timesteps,
              "sigma_schedule": sigma_schedule, "denoiser_items": adapter.denoiser_items,
              "sampling_seconds": sampling_seconds, "end_to_end_seconds": time.perf_counter() - started,
              "denoiser_seconds": denoiser_seconds,
              "agent_seconds": (agent.seconds - agent_before) if agent else 0.0,
              "agent_calls": (agent.count - decisions_before) if agent else 0,
              "final_policy_version": programs.version, "terminal_sigma": sigma,
              "failure_count": failure_count, "recovered": bool(recovered_steps),
              "recovery_steps": recovered_steps, "default_fallback_steps": default_steps,
              "valid_for_benchmark": failure_count == 0, "retry_nfe": 0}
    result["method"] = method
    if method == "grid_skip":
        result.update(reference_sigmas=list(sigmas), kept_indices=kept_indices,
                      skipped_indices=skipped_indices, grid_checks=grid_checks,
                      terminal_index=len(sigmas) - 1)
    return image, result, history
