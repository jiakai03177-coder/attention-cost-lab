import unittest

from attention_cost_lab.core import AttentionConfig, estimate_attention, format_bytes, optimization_notes


class CoreTests(unittest.TestCase):
    def test_kv_cache_bytes_for_small_shape(self):
        estimate = estimate_attention(
            AttentionConfig(
                layers=2,
                hidden_size=8,
                heads=2,
                kv_heads=1,
                seq_len=4,
                batch_size=1,
                dtype="fp16",
            )
        )

        self.assertEqual(estimate.head_dim, 4)
        self.assertEqual(estimate.group_size, 2)
        self.assertEqual(estimate.kv_cache_bytes, 2 * 4 * 1 * 4 * 2 * 2)
        self.assertEqual(estimate.kv_cache_growth_per_token_bytes, 2 * 1 * 4 * 2 * 2)


    def test_prefill_and_decode_attention_flops(self):
        estimate = estimate_attention(
            AttentionConfig(layers=1, hidden_size=8, heads=2, kv_heads=2, seq_len=4, dtype="fp16")
        )

        self.assertEqual(estimate.prefill_attention_flops, 4 * 1 * 1 * 2 * (4**2) * 4)
        self.assertEqual(estimate.decode_attention_flops_per_token, 4 * 1 * 1 * 2 * 4 * 4)


    def test_gqa_note_is_reported(self):
        estimate = estimate_attention(
            AttentionConfig(layers=32, hidden_size=4096, heads=32, kv_heads=8, seq_len=4096)
        )

        notes = optimization_notes(estimate)
        self.assertTrue(any("GQA is active" in note for note in notes))
        self.assertTrue(any("KV cache" in note or "KV cache bandwidth" in note for note in notes))


    def test_format_bytes(self):
        self.assertEqual(format_bytes(1024), "1.00 KiB")
        self.assertEqual(format_bytes(1024 * 1024), "1.00 MiB")


if __name__ == "__main__":
    unittest.main()
