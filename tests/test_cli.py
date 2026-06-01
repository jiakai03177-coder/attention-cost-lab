import json
import tempfile
import unittest
from pathlib import Path

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

    def test_cli_roofline_json_output(self):
        from contextlib import redirect_stdout
        from io import StringIO

        output = StringIO()
        with redirect_stdout(output):
            code = main([
                "--preset",
                "llama3-8b",
                "--seq-len",
                "2048",
                "--memory-bandwidth-gbps",
                "1000",
                "--compute-tflops",
                "100",
                "--json",
            ])

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["roofline"]["memory_bandwidth_gbps"], 1000)
        self.assertEqual(payload["roofline"]["compute_tflops"], 100)
        self.assertEqual(payload["roofline"]["bottleneck"], "memory")
        self.assertGreater(payload["roofline"]["decode_tokens_per_second"], 0)

    def test_cli_sweep_roofline_table_output(self):
        from contextlib import redirect_stdout
        from io import StringIO

        output = StringIO()
        with redirect_stdout(output):
            code = main([
                "--preset",
                "llama3-8b",
                "--sweep",
                "1024,2048",
                "--memory-bandwidth-gbps",
                "1000",
                "--compute-tflops",
                "100",
            ])

        text = output.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("decode roofline", text)
        self.assertIn("roofline throughput", text)
        self.assertIn("memory", text)

    def test_cli_writes_sweep_csv(self):
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "sweep.csv"
            output = StringIO()
            with redirect_stdout(output):
                code = main(["--preset", "llama3-8b", "--sweep", "1024,2048", "--csv", str(csv_path)])

            self.assertEqual(code, 0)
            text = csv_path.read_text(encoding="utf-8")
            self.assertIn("seq_len,batch_size,layers", text)
            self.assertIn("1024", text)
            self.assertIn("2048", text)
            self.assertIn("kv_cache_bytes", text)

    def test_cli_json_remains_machine_readable_with_csv(self):
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "single.csv"
            output = StringIO()
            with redirect_stdout(output):
                code = main([
                    "--preset",
                    "llama3-8b",
                    "--seq-len",
                    "1024",
                    "--json",
                    "--csv",
                    str(csv_path),
                ])

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["config"]["seq_len"], 1024)
            self.assertTrue(csv_path.exists())

    def test_cli_writes_roofline_csv(self):
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "roofline.csv"
            output = StringIO()
            with redirect_stdout(output):
                code = main([
                    "--preset",
                    "llama3-8b",
                    "--sweep",
                    "1024,2048",
                    "--memory-bandwidth-gbps",
                    "1000",
                    "--compute-tflops",
                    "100",
                    "--csv",
                    str(csv_path),
                ])

            self.assertEqual(code, 0)
            text = csv_path.read_text(encoding="utf-8")
            self.assertIn("decode_roofline_seconds_per_step", text)
            self.assertIn("roofline_bottleneck", text)
            self.assertIn("memory", text)


if __name__ == "__main__":
    unittest.main()
