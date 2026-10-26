# Failure recovery

Implementation reference: OPAL commit `8b16759a3e9930916ed582e28068267859f935f8`,
especially `opal/misc/recovery.py`, `opal/agent.py`, and `opal/report.py`.
This implementation adapts the version/trial/reporting ideas to deterministic SD3
sampling; it does not change the numerical solver or base model.

## Step transaction

Each denoiser prediction is computed once. Analysis and action code receive isolated
copies of the current/past state. Their memory is committed only after the action
passes trusted sigma/budget validation and the official Scheduler produces a finite
latent with the original shape, dtype, and device. Current Scheduler coordinates
and indices are restored before retrying a failed attempt. RNG consumed by a failed
attempt is restored. Earlier committed steps are retained.

The full policy pair (analysis, updater, relative helpers) is versioned and reverted
together. Mutable policy state belongs in `memory`, not module globals. A different
fallback version starts with fresh memory, including at a noninitial timestep.
The failure ledger explicitly records this reset; it is not an exact replay of the
trajectory that the fallback would have generated from the original image seed.

A version becomes eligible as a fallback after two successful committed steps.
Replacing a version before it passes this trial does not promote it. Restoration
tries eligible versions from newest to oldest, excluding failed versions. If those
fail, an immutable built-in policy follows the remaining official reference sigma
schedule. Failure of that policy, a model failure, or a damaged CUDA context aborts
the image and records the available diagnostic snapshot.

## Agent and mailbox

Protocol 2 responses bind `request_id`, `base_version`, a decision-local candidate
directory, and a candidate content hash. The bridge never writes a running policy
workspace in this protocol. The sampler verifies identity, content, imports, current
observations/actions and the numerical update before publishing the workspace.
Cancelled requests cannot be reopened. Late responses remain isolated in their
decision directories. Protocol 1 is retained only for existing deployments.

CLI timeout/failure/incomplete JSONL turns never count as successful edits. The CLI
and its tool process group are terminated on timeout; partial edits are retained
under the decision's `rejected/` directory when the bridge can reach the mailbox.
Diagnostic failures return immediately to the Agent so it may repair the probe at
the same paused step. Ordinary exceptions and `SystemExit` are recoverable policy
errors; `KeyboardInterrupt` is recorded and propagated.
On the Linux sampler's main thread, policy imports, analysis and action calls also
have a Python signal deadline (default 30 seconds; the sampling config can set
`policy_timeout_seconds`). Ordinary Python loops are interrupted and recovered.

## Reports and evaluation

- `failures.jsonl`: error stage, traceback, sample/seed, step/NFE/sigma/timestep,
  failed/target/effective versions, current-step restoration, memory reset, actual
  continuation, extra NFE and wall time from failure detection to successful step.
- `failure_report.md`: generated from the same records, without an LLM call.
- `decisions/<id>/outcome.json`: applied, kept, rejected, or later reverted.
- `trajectories/<sample>/failure.json` and, when possible, `failure-state.pt`:
  aborted-image metadata and trusted diagnostic state. Automatic restart of a killed
  Slurm job from this snapshot is not implemented.
- The next Agent request contains `previous_answer` and recent failures, with the
  complete ledger/report available in the run directory. Diagnostic errors are also
  present in the immediate inspection response.

Per-image results include `failure_count`, `recovered`, `recovery_steps`,
`default_fallback_steps`, `retry_nfe`, and `valid_for_benchmark`. No denoiser forward
is repeated for policy recovery. Total sampling/end-to-end latency includes failed
attempts. CUDA forward attempts are counted before execution.

Frozen samples that require recovery are explicitly invalid for the clean method
comparison. Results retain all images and a separate `clean_summary`; the paired
mini20 comparison refuses to silently pool recovered/default-fallback samples.
Quality degradation is feedback, not an automatic rollback threshold.

## Boundaries

This remains trusted in-process Python, not a security sandbox. Process death,
native crashes, uninterruptible native calls, hardware faults, and a corrupt CUDA
context cannot be made safe by catching Python exceptions. Signal deadlines cannot
preempt a native call that never returns and are only active on the Linux main
thread. Automatic Slurm-job resumption is not implemented. Disk/connection failures can prevent writing a
report; the implementation does not claim restoration without a valid state.

RunPod changes are not deployed to Trento until explicitly authorized. A bridge
already running from the previous code is not restarted by this source update.
