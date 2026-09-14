import importlib.util
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class LocalClientTest(unittest.TestCase):
    def test_seed_usage_and_expired_budget(self):
        self.assertIsNotNone(importlib.util.find_spec("local_client"))
        from local_client import LocalClient

        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"model": "actor", "choices": [{"message": {"content": "ACTION: look"}}], "usage": {"prompt_tokens": 17, "completion_tokens": 3}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = LocalClient(f"http://127.0.0.1:{server.server_port}/v1", "actor", 19, time.time() + 10)
            result = client.call("goal", tag="step0")
            self.assertTrue(result.ok)
            self.assertEqual(result.text, "ACTION: look")
            self.assertEqual(result.usage_vec()["input_tokens"], 17)
            self.assertEqual(result.models, ["actor"])
            self.assertEqual(requests[0]["messages"][1]["content"], "goal")
            client.call("goal", tag="step1")
            self.assertNotEqual(requests[0]["seed"], requests[1]["seed"])
            self.assertEqual(sum(event["input_tokens"] for event in client.events), 34)
            self.assertEqual(len(client.events), 2)
            same = LocalClient(client.endpoint, "actor", 19, time.time() + 10)
            same.call("goal", tag="step0")
            self.assertEqual(requests[0]["seed"], requests[2]["seed"])
            deliberating = LocalClient(client.endpoint, "actor", 20, time.time() + 10, system_prompt="Think then act", max_tokens=256)
            deliberating.call("goal")
            self.assertEqual(requests[3]["messages"][0]["content"], "Think then act")
            self.assertEqual(requests[3]["max_tokens"], 256)
            expired = LocalClient(client.endpoint, "actor", 19, time.time() - 1)
            self.assertFalse(expired.call("goal").ok)
            self.assertEqual(len(requests), 4)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
