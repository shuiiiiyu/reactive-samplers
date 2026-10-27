"""Method 2: decide on the CURRENT node before prediction; skip omits its entire update."""
import time
from .context import Context, Prediction, validate_skip
from .models.cache import prepare_feature, refresh_prediction
from .programs import digest
from .recovery import (FailureLog, POLICY_ERRORS, PolicyRecovery, clone_state, fatal_error,
                       policy_deadline, require_finite, restore_rng, rng_state, save_crash)
from .wrapper import _analyze


def sample_grid_nodes(adapter, programs, prompt, seed, reference_steps, max_nfe, schedule, recorder, agent=None,
           policy_timeout_seconds=30):
    method = "grid_skip"
    if programs.method != method:
        raise ValueError("Sampling method and policy method must match")
    if max_nfe < min(2, reference_steps) or reference_steps < 1:
        raise ValueError("grid_skip needs a positive grid and enough NFE for its protected first/last nodes")
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
    kept_indices, skipped_indices = [], []
    nfe = consecutive_skips = 0
    reference_latent = reference_feature = reference_sigma = reference_step = None
    agent_before = agent.seconds if agent else 0.0
    decisions_before = agent.count if agent else 0
    try:
        latent, sigmas = adapter.start(prompt, seed, reference_steps)
        sigma, terminal = sigmas[0], sigmas[-1]
        sigma_schedule.append(sigma)
        sampling_started = time.perf_counter()
        for step, (sigma, interval_end) in enumerate(zip(sigmas, sigmas[1:])):
            stage = "input_feature"
            feature = prepare_feature(adapter, latent, sigma)
            ctx = Context(prompt, seed, step, nfe, max_nfe, sigma, terminal,
                          adapter.timestep(sigma), tuple(sigmas), latent, Prediction(None), history,
                          sample_id=recorder.root.name, input_feature=feature,
                          reference_feature=reference_feature, reference_latent=reference_latent,
                          observation_reference_step=reference_step, observation_reference_sigma=reference_sigma,
                          cache_observation={"sigma": sigma, "step_delta_sigma": sigma-interval_end,
                              "since_real_delta_sigma": reference_sigma-sigma if reference_sigma is not None else 0.,
                              "consecutive_skips": consecutive_skips})
            ctx.method = method
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
                    if getattr(programs.analysis, "FEATURE_LAYERS", []):
                        raise ValueError("Current-node decisions allow only shallow input features")
                    break
                except POLICY_ERRORS as exc:
                    revert(exc, stage)
            prediction = None
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
                    if getattr(programs.analysis, "FEATURE_LAYERS", []):
                        raise ValueError("Current-node decisions allow only shallow input features")
                    stage = "analysis"
                    if analyzed is None:
                        analyzed = _analyze(programs, ctx, memory, policy_timeout_seconds)
                    stage = "updater"
                    observation, proposed_memory, working = analyzed
                    with policy_deadline(policy_timeout_seconds):
                        action = programs.updater.choose(clone_state(observation), working, proposed_memory)
                    skip = validate_skip(action)
                    forced = None
                    if step == 0 or step == len(sigmas)-2:
                        skip, forced = False, "initialization" if step == 0 else "terminal_interval"
                    elif nfe >= max_nfe-1:
                        skip, forced = True, "budget"
                    proposed_memory = clone_state(proposed_memory)
                    next_sigma = interval_end
                    next_latent = latent
                    if not skip:
                        if prediction is None:
                            stage = "prediction"
                            adapter.synchronize()
                            prediction_started = time.perf_counter()
                            prediction, _ = refresh_prediction(adapter, latent, sigma, feature, residual=False)
                            adapter.synchronize()
                            denoiser_seconds += time.perf_counter()-prediction_started
                            nfe += 1
                            ctx.nfe = nfe
                            require_finite(prediction.guided, "model prediction")
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
            if not skip:
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
            record.update(skipped_node=skip, requested_skip=action.skip, forced=forced,
                          backbone_evaluated=prediction is not None, scheduler_updated=not skip,
                          grid_decisions=[{"index": step, "sigma": sigma, "timestep": ctx.timestep,
                                           "skip": skip, "forced": forced}])
            if decision_status is not None:
                record.update(agent_decision=agent.count, agent_outcome=decision_status,
                              agent_output=str(agent.current_folder / "answer.txt"))
            if skip:
                skipped_indices.append(step)
                consecutive_skips += 1
            else:
                kept_indices.append(step)
                consecutive_skips = 0
                reference_latent, reference_feature = latent, feature
                reference_sigma, reference_step = sigma, step
            if agent:
                agent.record_execution(ctx, record)
            recorder.event("step", **record)
            timesteps.append(ctx.timestep)
            sigma_schedule.append(next_sigma)
            history.append({**record, "latent": latent,
                            "prediction": clone_state(prediction) if prediction is not None else Prediction(None)})
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
    result = {"nfe": nfe, "steps": step + 1, "timesteps": timesteps,
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
    result.update(reference_sigmas=list(sigmas), kept_indices=kept_indices,
                  skipped_indices=skipped_indices, grid_checks=len(sigmas)-1,
                  scheduler_updates=len(kept_indices), feature_extractions=len(sigmas)-1,
                  terminal_index=len(sigmas) - 1)
    return image, result, history
