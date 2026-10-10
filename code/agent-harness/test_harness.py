"""Tests for harness.py that never touch the network: python3 -m unittest test_harness"""
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import harness


class FakeDecisions:
    def __init__(self, risk=0.1, authorized=0.9, error=None):
        self.risk, self.authorized, self.error, self.calls = risk, authorized, error, 0

    def create(self, **request):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(answers=[
            SimpleNamespace(name="risk", type="score", score=self.risk),
            SimpleNamespace(name="authorized", type="predicate", probability=self.authorized),
        ])


def fake_client(**answers):
    return SimpleNamespace(decisions=FakeDecisions(**answers))


class FakeModel:
    """Replays scripted replies and records what it was sent."""

    def __init__(self, replies):
        self.replies, self.sent, self.cost = list(replies), [], 0.0

    def complete(self, messages, tools):
        self.sent.append([dict(m) for m in messages])
        return self.replies.pop(0)


def tool_call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call_{name}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


class ExtensionSurfacesTest(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        (self.dir / "AGENTS.md").write_text("Use Australian English.")
        for name, description in [("good", "Does <things> & more."), ("empty", "")]:
            skill = self.dir / ".agents/skills" / name
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\nSteps.")

    def test_context_is_wrapped_with_its_path(self):
        context = harness.find_context(self.dir)
        self.assertIn(f'<project_instructions path="{self.dir / "AGENTS.md"}">', context)
        self.assertIn("Use Australian English.", context)

    def test_skills_without_a_description_are_skipped(self):
        self.assertEqual([s["name"] for s in harness.find_skills(self.dir)], ["good"])

    def test_skill_text_is_escaped(self):
        prompt = harness.format_skills(harness.find_skills(self.dir))
        self.assertIn("Does &lt;things&gt; &amp; more.", prompt)
        self.assertIn("<available_skills>", prompt)


class ToolsTest(unittest.TestCase):
    def test_schema_lists_parameters(self):
        params = harness.schema("read_file", harness.tool_read_file)["function"]["parameters"]
        self.assertEqual(params["properties"]["offset"], {"type": "integer"})
        self.assertEqual(params["required"], ["path"])

    def test_read_file_numbers_lines_from_the_offset(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("a\nb\nc\n")
        self.assertEqual(harness.tool_read_file(f.name, offset=1, limit=1), "   2: b")


class SafetyTest(unittest.TestCase):
    def decide(self, client, tool="bash"):
        return harness.classify_tool_call(client, "Summarise data/sales.csv", tool, {"cmd": "ls"})[0]

    def test_read_only_tools_skip_the_api(self):
        client = fake_client(error=AssertionError("should not be called"))
        self.assertEqual(self.decide(client, tool="read_file"), "allow")

    def test_thresholds(self):
        self.assertEqual(self.decide(fake_client(risk=0.2, authorized=0.9)), "allow")
        self.assertEqual(self.decide(fake_client(risk=0.2, authorized=0.3)), "ask")
        self.assertEqual(self.decide(fake_client(risk=1.4, authorized=0.9)), "ask")
        self.assertEqual(self.decide(fake_client(risk=2.8, authorized=0.9)), "deny")

    def test_api_failure_falls_back_to_asking(self):
        self.assertEqual(self.decide(fake_client(error=TimeoutError())), "ask")

    def test_nobody_at_the_terminal_means_no(self):
        with mock.patch("sys.stdin", SimpleNamespace(isatty=lambda: False)):
            self.assertFalse(harness.ask_user("bash", {"cmd": "rm -rf build"}, "risky"))


class ContextTest(unittest.TestCase):
    def test_compaction_keeps_tool_results_with_their_calls(self):
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        for i in range(5):
            messages += [tool_call("bash", cmd=f"step {i}"),
                         {"role": "tool", "tool_call_id": "call_bash", "content": "ok"}]
        model = FakeModel([{"role": "assistant", "content": "did five steps"}])
        compacted = harness.compact(messages, model, keep=3)
        self.assertEqual(compacted[:2], messages[:2])
        self.assertIn("did five steps", compacted[2]["content"])
        self.assertNotEqual(compacted[3]["role"], "tool")


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())

    def test_runs_a_tool_and_returns_the_final_answer(self):
        model = FakeModel([tool_call("bash", cmd="echo hi"), {"role": "assistant", "content": "It printed hi."}])
        answer = harness.run("Say hi", self.dir, model, fake_client())
        self.assertEqual(answer, "It printed hi.")
        tool_result = model.sent[1][-1]
        self.assertEqual(tool_result["role"], "tool")
        self.assertIn("stdout:\nhi", tool_result["content"])

    def test_a_denied_call_is_not_run(self):
        model = FakeModel([tool_call("bash", cmd="echo should-not-run"), {"role": "assistant", "content": "ok"}])
        harness.run("Say hi", self.dir, model, fake_client(risk=2.9))
        self.assertIn("BLOCKED", model.sent[1][-1]["content"])
        self.assertNotIn("should-not-run\n", model.sent[1][-1]["content"])

    def test_stops_at_the_turn_limit(self):
        model = FakeModel([tool_call("read_file", path="nope")] * 3)
        self.assertEqual(harness.run("Loop forever", self.dir, model, fake_client(), max_turns=3),
                         "Stopped: hit 3 turns.")


if __name__ == "__main__":
    unittest.main()
