import json
import unittest

from attention_cost_lab.cli import main


class CliTests(unittest.TestCase):
    def test_cli_json_output(self):
        from contextlib import redirect_stdout
        from io import StringIO

        output = StringIO()
        with redirect_stdout(output):
            code = main(["--preset", "llama3-8b", "--seq-len", "2048", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["config"]["layers"], 32)
        self.assertEqual(payload["config"]["kv_heads"], 8)
        self.assertEqual(payload["estimate"]["head_dim"], 128)
        self.assertTrue(payload["notes"])

    def test_cli_sweep_json_output(self):
        from contextlib import redirect_stdout
        from io import StringIO

        output = StringIO()
        with redirect_stdout(output):
            code = main(["--preset", "llama3-8b", "--sweep", "1024,2048,4096", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual([item["config"]["seq_len"] for item in payload["sweep"]], [1024, 2048, 4096])
        self.assertLess(
            payload["sweep"][0]["estimate"]["kv_cache_bytes"],
            payload["sweep"][2]["estimate"]["kv_cache_bytes"],
        )

    def test_cli_sweep_table_output(self):
        from contextlib import redirect_stdout
        from io import StringIO

        output = StringIO()
        with redirect_stdout(output):
            code = main(["--preset", "llama3-8b", "--sweep", "1024,2048"])

        text = output.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Context Sweep", text)
        self.assertIn("1024", text)
        self.assertIn("2048", text)


if __name__ == "__main__":
    unittest.main()
