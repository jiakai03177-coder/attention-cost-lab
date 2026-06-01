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


if __name__ == "__main__":
    unittest.main()
