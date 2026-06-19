import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.signal_processing.agent_signal_processing import _tool_control_hook


class StubLLM:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="Final report from existing results.")


class SignalToolControlTests(unittest.TestCase):
    def test_repairs_alias_and_missing_id_below_budget(self):
        llm = StubLLM()
        pending = AIMessage(
            content="",
            id="pending",
            tool_calls=[{"name": "compute_all", "args": {"ticker": "AMZN"}, "id": ""}],
        )

        result = asyncio.run(
            _tool_control_hook(llm, {"compute_all_metrics"}, 2)(
                {"messages": [HumanMessage(content="Analyze AMZN"), pending]}
            )
        )

        repaired = result["messages"][0]
        self.assertEqual(repaired.tool_calls[0]["name"], "compute_all_metrics")
        self.assertTrue(repaired.tool_calls[0]["id"].startswith("autofixed-"))
        self.assertEqual(llm.calls, [])

    def test_forces_plain_final_response_at_budget(self):
        llm = StubLLM()
        messages = [HumanMessage(content="Analyze AMZN")]
        for index in range(2):
            messages.extend(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "compute_all_metrics",
                                "args": {"ticker": "AMZN"},
                                "id": f"call-{index}",
                            }
                        ],
                    ),
                    ToolMessage(content="{}", tool_call_id=f"call-{index}"),
                ]
            )
        messages.append(
            AIMessage(
                content="",
                id="pending",
                tool_calls=[
                    {
                        "name": "compute_all_metrics",
                        "args": {"ticker": "AMZN"},
                        "id": "call-pending",
                    }
                ],
            )
        )

        result = asyncio.run(
            _tool_control_hook(llm, {"compute_all_metrics"}, 2)({"messages": messages})
        )

        final = result["messages"][0]
        self.assertEqual(final.id, "pending")
        self.assertEqual(final.content, "Final report from existing results.")
        self.assertEqual(final.tool_calls, [])
        self.assertEqual(len(llm.calls), 1)
        self.assertNotIn(messages[-1], llm.calls[0])

    def test_budget_resets_at_latest_user_message(self):
        llm = StubLLM()
        old_call = AIMessage(
            content="",
            tool_calls=[
                {"name": "compute_all_metrics", "args": {}, "id": "old-call"}
            ],
        )
        pending = AIMessage(
            content="",
            id="pending",
            tool_calls=[
                {"name": "compute_all_metrics", "args": {}, "id": "new-call"}
            ],
        )
        messages = [
            HumanMessage(content="Old request"),
            old_call,
            ToolMessage(content="{}", tool_call_id="old-call"),
            AIMessage(content="Old final response"),
            HumanMessage(content="New request"),
            pending,
        ]

        result = asyncio.run(
            _tool_control_hook(llm, {"compute_all_metrics"}, 1)({"messages": messages})
        )

        self.assertEqual(result, {})
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
