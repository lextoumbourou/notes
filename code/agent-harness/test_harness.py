"""Tests for harness.py that never touch the network: python3 -m unittest test_harness"""
import json
import pathlib
import tempfile
import unittest
from functools import partial
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


def fake_policy(**answers):
    return partial(harness.classify_tool_call, fake_client(**answers))


class FakeModel:
    """Replays scripted replies and records what it was sent."""

    def __init__(self, replies):
        self.replies, self.sent, self.cost = list(replies), [], 0.0

    def complete(self, messages, tools):
        self.sent.append([dict(m) for m in messages])
        return self.replies.pop(0)


def tool_call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call_{name}", "name": name, "arguments": json.dumps(args)}]}


class ExtensionSurfacesTest(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        (self.dir / "AGENTS.md").write_text(
            "Use Australian English.\nPrefer <tools> & notes."
        )
        for name, description in [("good", "Does <things> & more."), ("empty", "")]:
            skill = self.dir / ".agents/skills" / name
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\nSteps.")

    def test_context_is_wrapped_with_its_path(self):
        context = harness.find_context(self.dir)
        self.assertIn(f'<project_instructions path="{self.dir / "AGENTS.md"}">', context)
        self.assertIn("Use Australian English.", context)
        self.assertIn("Prefer &lt;tools&gt; &amp; notes.", context)

    def test_skills_without_a_description_are_skipped(self):
        self.assertEqual([s["name"] for s in harness.find_skills(self.dir)], ["good"])

    def test_skill_text_is_escaped(self):
        prompt = harness.format_skills(harness.find_skills(self.dir))
        self.assertIn("Does &lt;things&gt; &amp; more.", prompt)
        self.assertIn("<available_skills>", prompt)


class ToolsTest(unittest.TestCase):
    def test_schema_lists_parameters(self):
        params = harness.schema("read_file", harness.tool_read_file)["parameters"]
        self.assertEqual(params["properties"]["offset"], {"type": "integer"})
        self.assertEqual(params["required"], ["path"])

    def test_read_file_numbers_lines_from_the_offset(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("a\nb\nc\n")
        self.assertEqual(harness.tool_read_file(f.name, offset=1, limit=1,
                                                base_dir=pathlib.Path(f.name).parent), "   2: b")

    def test_file_tools_reject_paths_outside_the_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            work = root / "work"
            work.mkdir()
            outside = root / "private.txt"
            outside.write_text("private")
            (work / "link.txt").symlink_to(outside)
            for path in ("../private.txt", "link.txt"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    harness.tool_read_file(path, base_dir=work)
            with self.assertRaises(ValueError):
                harness.tool_write_file("../private.txt", "oops", base_dir=work)
            self.assertEqual(outside.read_text(), "private")

    def test_bash_runs_from_the_supplied_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = harness.tool_bash("pwd", base_dir=pathlib.Path(tmp))
            self.assertIn(f"stdout:\n{pathlib.Path(tmp).resolve()}\n", result)

    def test_search_replace_rejects_empty_search(self):
        with self.assertRaises(ValueError):
            harness.tool_search_replace("file.txt", "", "new")


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
        self.assertEqual(model.sent[0][1]["content"], "task")


class ModelAdapterTest(unittest.TestCase):
    def test_chat_completions_adapter_translates_tools_and_replies(self):
        api_call = SimpleNamespace(id="call_1", function=SimpleNamespace(
            name="read_file", arguments='{"path":"data.txt"}'))
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[api_call]))],
        )
        create = mock.Mock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        model = harness.ChatCompletionsModel(client)
        messages = [{"role": "user", "content": "Read data"},
                    tool_call("read_file", path="data.txt"),
                    {"role": "tool", "tool_call_id": "call_read_file", "content": "data"}]
        reply = model.complete(messages, [harness.schema("read_file", harness.tool_read_file)])
        request = create.call_args.kwargs
        self.assertEqual(request["messages"][1]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(request["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(reply["tool_calls"][0]["name"], "read_file")
        self.assertAlmostEqual(model.cost, 0.00002)


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())

    def test_runs_a_tool_and_returns_the_final_answer(self):
        model = FakeModel([tool_call("bash", cmd="echo hi"), {"role": "assistant", "content": "It printed hi."}])
        answer = harness.run("Say hi", self.dir, model, fake_policy())
        self.assertEqual(answer, "It printed hi.")
        tool_result = model.sent[1][-1]
        self.assertEqual(tool_result["role"], "tool")
        self.assertIn("stdout:\nhi", tool_result["content"])

    def test_accepts_a_policy_without_an_openai_client(self):
        model = FakeModel([tool_call("bash", cmd="echo hi"), {"role": "assistant", "content": "done"}])
        policy = mock.Mock(return_value=("allow", "test policy"))
        self.assertEqual(harness.run("Say hi", self.dir, model, policy), "done")
        policy.assert_called_once()

    def test_a_denied_call_is_not_run(self):
        model = FakeModel([tool_call("bash", cmd="echo should-not-run"), {"role": "assistant", "content": "ok"}])
        harness.run("Say hi", self.dir, model, fake_policy(risk=2.9))
        self.assertIn("BLOCKED", model.sent[1][-1]["content"])
        self.assertNotIn("should-not-run\n", model.sent[1][-1]["content"])

    def test_stops_at_the_turn_limit(self):
        model = FakeModel([tool_call("read_file", path="nope")] * 3)
        self.assertEqual(harness.run("Loop forever", self.dir, model, fake_policy(), max_turns=3),
                         "Stopped: hit 3 turns.")

    def test_malformed_tool_arguments_are_reported_to_the_model(self):
        bad_call = tool_call("read_file", path="data.txt")
        bad_call["tool_calls"][0]["arguments"] = "{bad json"
        model = FakeModel([bad_call, {"role": "assistant", "content": "fixed"}])
        self.assertEqual(harness.run("Read data", self.dir, model, fake_policy()), "fixed")
        self.assertIn("invalid tool arguments", model.sent[1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
