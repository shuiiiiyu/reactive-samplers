"""Only trusted, explicitly configured agent commands are launched."""
import subprocess


class KeepAgent:
    def __call__(self, request_path, workspace):
        return "keep"


class CommandAgent:
    def __init__(self, argv):
        if not argv or not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
            raise ValueError("agent.command must be a nonempty argument list")
        self.argv = argv

    def __call__(self, request_path, workspace):
        # The bridge owns CLI/provider session resumption. No shell interpolation.
        result = subprocess.run(
            [s.replace("{request}", str(request_path)).replace("{workspace}", str(workspace)) for s in self.argv],
            cwd=workspace, text=True, capture_output=True, check=True,
        )
        return result.stdout
