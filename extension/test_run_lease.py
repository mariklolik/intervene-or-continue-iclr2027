import json
import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class LeaseReadinessTest(unittest.TestCase):
    def test_live_endpoint_requires_exact_model_and_available_time(self):
        from run_lease import await_model
        model = ["wrong"]

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"data": [{"id": model[0]}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        url = f"http://127.0.0.1:{server.server_port}/v1"
        try:
            with self.assertRaises(RuntimeError):
                await_model(url, "actor", time.time() + 0.1, child)
            model[0] = "actor"
            await_model(url, "actor", time.time() + 1, child)
            with self.assertRaises(RuntimeError):
                await_model(url, "actor", time.time() - 1, child)
            child.terminate()
            child.wait()
            with self.assertRaises(RuntimeError):
                await_model(url, "actor", time.time() + 1, child)
        finally:
            child.terminate()
            child.wait()
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
