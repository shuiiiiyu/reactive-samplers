"""One image's control loop. Model prediction and numerical updates stay outside policies."""
import copy
import time

from .context import Context, validate_action


def sample(adapter, programs, prompt, seed, reference_steps, max_nfe, schedule, recorder, agent=None):
    if max_nfe < 1 or reference_steps < 1:
        raise ValueError("reference_steps and max_nfe must be positive")
    started = time.perf_counter()
    latent, sigmas = adapter.start(prompt, seed, reference_steps)
    sigma, terminal = sigmas[0], sigmas[-1]
    history, memory = [], {}  # program persists across images; state does not
    agent_before = agent.seconds if agent else 0.0
    sampling_started = time.perf_counter()
    for step in range(max_nfe):
        layers = getattr(programs.analysis, "FEATURE_LAYERS", [])
        prediction = adapter.predict(latent, sigma, layers)
        ctx = Context(prompt, seed, step, step + 1, max_nfe, sigma, terminal,
                      adapter.timestep(sigma), tuple(sigmas), latent, prediction, history)
        ctx.analysis_memory = copy.deepcopy(memory)
        observation = programs.analysis.analyze(ctx, memory)
        selected_action = None
        if agent and schedule.due(step):
            old = (programs.analysis, programs.updater, programs.version)
            previous_memory = copy.deepcopy(memory)
            previous_observation = observation
            try:
                agent.check(ctx, observation, programs)
                changed = programs.reload()
                # A changed probe is rerun at this same paused step, before choosing its action.
                if changed:
                    candidate_memory = copy.deepcopy(ctx.analysis_memory)
                    observation = programs.analysis.analyze(ctx, candidate_memory)
                    selected_action = programs.updater.choose(observation, ctx, candidate_memory)
                    validate_action(selected_action, ctx)
                    memory = candidate_memory
                recorder.event("decision", step=step, nfe=ctx.nfe, version=programs.version, changed=changed)
            except Exception as exc:
                programs.analysis, programs.updater, programs.version = old
                programs.restore_workspace()
                memory = previous_memory
                selected_action = None
                observation = previous_observation
                recorder.event("decision_rejected", step=step, error=str(exc), version=programs.version)
        action = selected_action if selected_action is not None else programs.updater.choose(observation, ctx, memory)
        next_sigma = validate_action(action, ctx)
        # No policy gets the scheduler/model object or writes the latent update.
        next_latent = adapter.advance(latent, prediction, sigma, next_sigma)
        record = {"step": step, "nfe": ctx.nfe, "sigma": sigma, "next_sigma": next_sigma,
                  "version": programs.version, "observation": observation}
        recorder.event("step", **record)
        history.append({**record, "latent": latent, "prediction": prediction})
        if agent is None:
            limit = getattr(programs.analysis, "HISTORY_LIMIT", 2)
            if limit is not None:
                if type(limit) is not int or limit < 0:
                    raise ValueError("HISTORY_LIMIT must be a nonnegative integer or None")
                del history[:max(0, len(history) - limit)]
        latent, sigma = next_latent, next_sigma
        if sigma == terminal:
            break
    adapter.synchronize()
    sampling_seconds = time.perf_counter() - sampling_started
    image = adapter.finish(latent)
    adapter.synchronize()
    result = {"nfe": step + 1, "denoiser_items": adapter.denoiser_items,
              "sampling_seconds": sampling_seconds, "end_to_end_seconds": time.perf_counter() - started,
              "agent_seconds": (agent.seconds - agent_before) if agent else 0.0,
              "final_policy_version": programs.version, "terminal_sigma": sigma}
    return image, result, history
