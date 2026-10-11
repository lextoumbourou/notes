# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=3.26"]
# ///
"""Run the example harness while recording its complete model and tool exchange.

Usage: uv run trace_run.py --web-search --trace /path/to/trace.jsonl WORKDIR "task"
The trace is written outside WORKDIR so the agent does not see its own log.
"""

import argparse
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone
from functools import wraps
from types import SimpleNamespace

import openai

import harness


class Trace:
    def __init__(self, path: pathlib.Path):
        self.file = path.open("x", encoding="utf-8")

    def record(self, event: str, **data):
        json.dump(
            {"time": datetime.now(timezone.utc).isoformat(), "event": event, **data},
            self.file, ensure_ascii=False, default=str,
        )
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()


class RecordingResource:
    def __init__(self, resource, name: str, trace: Trace):
        self.resource, self.name, self.trace = resource, name, trace

    def create(self, **request):
        self.trace.record(f"{self.name}.request", request=request)
        try:
            response = self.resource.create(**request)
        except Exception as error:
            self.trace.record(f"{self.name}.error", type=type(error).__name__, message=str(error))
            raise
        self.trace.record(
            f"{self.name}.response",
            response=response.model_dump(mode="json", exclude_none=True),
        )
        return response


def record_tools(trace: Trace):
    for name, tool in harness.TOOLS.items():
        @wraps(tool)
        def recorded(*args, _name=name, _tool=tool, **kwargs):
            trace.record("tool.call", name=_name, args=args, kwargs=kwargs)
            try:
                result = _tool(*args, **kwargs)
            except Exception as error:
                trace.record("tool.error", name=_name, type=type(error).__name__, message=str(error))
                raise
            trace.record("tool.result", name=_name, result=result)
            return result

        harness.TOOLS[name] = recorded


def record_bash_process(trace: Trace):
    """Keep full shell output even when the tool result is truncated for the model."""
    original_run = subprocess.run

    def recorded_run(*args, **kwargs):
        command = args[0] if args else kwargs.get("args")
        try:
            result = original_run(*args, **kwargs)
        except subprocess.TimeoutExpired as error:
            trace.record("bash.process_timeout", command=command,
                         stdout=error.stdout, stderr=error.stderr)
            raise
        trace.record("bash.process", command=command, returncode=result.returncode,
                     stdout=result.stdout, stderr=result.stderr)
        return result

    harness.subprocess = SimpleNamespace(run=recorded_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-search", action="store_true")
    parser.add_argument("--trace", required=True, type=pathlib.Path)
    parser.add_argument("working_dir", type=pathlib.Path)
    parser.add_argument("task", nargs="+")
    args = parser.parse_args()
    task = " ".join(args.task)
    trace = Trace(args.trace)
    client = openai.OpenAI(default_headers={"Accept-Encoding": "gzip"})
    recording_client = SimpleNamespace(
        responses=RecordingResource(client.responses, "responses", trace),
        decisions=RecordingResource(client.decisions, "decisions", trace),
    )
    model = harness.ResponsesModel(recording_client, web_search=args.web_search)
    record_tools(trace)
    record_bash_process(trace)

    def policy(request, name, tool_args):
        decision, reason = harness.classify_tool_call(recording_client, request, name, tool_args)
        trace.record("safety.decision", name=name, args=tool_args,
                     decision=decision, reason=reason)
        return decision, reason

    original_ask = harness.ask_user

    def ask(name, tool_args, reason):
        approved = original_ask(name, tool_args, reason)
        trace.record("human.approval", name=name, args=tool_args,
                     reason=reason, approved=approved)
        return approved

    harness.ask_user = ask
    trace.record("run.start", task=task, working_dir=str(args.working_dir),
                 model=model.name, web_search=args.web_search)
    try:
        answer = harness.run(task, args.working_dir, model, policy)
        trace.record("run.answer", answer=answer)
        print(answer)
    except BaseException as error:
        trace.record("run.error", type=type(error).__name__, message=str(error))
        raise
    finally:
        trace.record("run.end", model_token_cost_usd=model.cost)
        print(f"cost=${model.cost:.4f}", file=sys.stderr)
        trace.close()
        client.close()


if __name__ == "__main__":
    main()
