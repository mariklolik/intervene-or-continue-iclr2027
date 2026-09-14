import json
import inspect
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class SamplingProbeTest(unittest.TestCase):
    def test_wrong_or_missing_context_rejects_before_any_call(self):
        from sampling_preflight import verify_sampling
        self.assertIn("expected_context", inspect.signature(verify_sampling).parameters)
        for cap in (None, 16384, 327680):
            with self.subTest(cap=cap), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                log = root / "server.log"
                text = "'enable_deterministic_inference': True, 'sampling_backend': 'pytorch'"
                log.write_text(text + (f", 'context_length': {cap}" if cap is not None else ""))
                with patch("sampling_preflight.LocalClient") as client:
                    with self.assertRaises(RuntimeError):
                        verify_sampling("http://127.0.0.1/v1", "actor", time.time() + 10, log, root / "probe.json", expected_context=32768)
                    client.assert_not_called()
                receipt = json.loads((root / "probe.json").read_text())
                self.assertEqual(receipt["expected_context"], 32768)
                self.assertFalse(receipt["context_log_match"])

    def test_long_context_requires_success_model_and_actual_token_usage(self):
        from sampling_preflight import verify_sampling
        self.assertIn("expected_context", inspect.signature(verify_sampling).parameters)
        cases = [(17001, True, "actor", True), (32767, True, "actor", True), (16384, True, "actor", False), (32768, True, "actor", False), (None, True, "actor", False), (17001, False, "actor", False), (17001, True, "wrong", False)]
        for tokens, ok, model, passed in cases:
            with self.subTest(tokens=tokens, ok=ok, model=model), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                log = root / "server.log"
                log.write_text("'enable_deterministic_inference': True, 'sampling_backend': 'pytorch', 'context_length': 32768")
                calls = []
                def make_client(*args, **kwargs):
                    long = kwargs["max_tokens"] == 1
                    usage = {"input_tokens": tokens if long else 5, "output_tokens": 1}
                    result = SimpleNamespace(ok=ok if long else True, text="x", models=[model if long else "actor"], usage=usage, errors=[])
                    client = Mock(events=[{"tag": "context-probe" if long else "seed-probe", **usage}])
                    client.call.return_value = result
                    calls.append((kwargs, client))
                    return client
                with patch("sampling_preflight.LocalClient", side_effect=make_client):
                    if passed:
                        verify_sampling("http://127.0.0.1/v1", "actor", time.time() + 10, log, root / "probe.json", expected_context=32768)
                    else:
                        with self.assertRaises(RuntimeError):
                            verify_sampling("http://127.0.0.1/v1", "actor", time.time() + 10, log, root / "probe.json", expected_context=32768)
                self.assertEqual([kwargs["max_tokens"] for kwargs, _ in calls], [64, 64, 64, 64, 1])
                calls[-1][1].call.assert_called_once_with(" x" * 17000, tag="context-probe")
                receipt = json.loads((root / "probe.json").read_text())
                self.assertEqual(receipt["passed"], passed)
                self.assertTrue(receipt["context_log_match"])
                self.assertEqual(len(receipt["events"]), 4)
                self.assertEqual(receipt["context_probe"]["usage"]["input_tokens"], tokens)
                self.assertEqual(len(receipt["context_probe"]["events"]), 1)

    def test_probe_rejects_seed_ignoring_server_and_preserves_events(self):
        from sampling_preflight import verify_sampling
        counter = [0]
        ignore = [False]

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                counter[0] += 1
                reply = str(counter[0] if ignore[0] else request["seed"])
                body = json.dumps({"model": "actor", "choices": [{"message": {"content": reply}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                log = root / "server.log"
                log.write_text("'enable_deterministic_inference': True, 'sampling_backend': 'pytorch'")
                endpoint = f"http://127.0.0.1:{server.server_port}/v1"
                verify_sampling(endpoint, "actor", time.time() + 10, log, root / "pass.json")
                self.assertTrue(json.loads((root / "pass.json").read_text())["passed"])
                ignore[0] = True
                with self.assertRaisesRegex(RuntimeError, "Sampling preflight"):
                    verify_sampling(endpoint, "actor", time.time() + 10, log, root / "fail.json")
                failed = json.loads((root / "fail.json").read_text())
                self.assertFalse(failed["passed"])
                self.assertEqual(len(failed["events"]), 4)
                log.write_text("'enable_deterministic_inference': False, 'sampling_backend': 'flashinfer'")
                before = counter[0]
                with self.assertRaisesRegex(RuntimeError, "Sampling preflight"):
                    verify_sampling(endpoint, "actor", time.time() + 10, log, root / "flag.json")
                self.assertEqual(counter[0], before)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
