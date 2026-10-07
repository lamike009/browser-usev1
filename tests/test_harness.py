"""Unit checks for the V1 pin, schema proof, and retry policy. Does not launch Chromium."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel, ValidationError

from agent_runner import (
    PINNED_BROWSER_USE,
    RETRY_BACKOFF_SECONDS,
    BlockedError,
    PlaceholderCard,
    PlaceholderExtract,
    RetryableStepError,
    RunResult,
    WorkerBusy,
    assert_placeholder_extract,
    classify_exception,
    installed_browser_use_version,
    parse_extract,
    run_step_with_retries,
    runtime_paths,
    make_browser,
    single_worker,
    task_extract,
    task_filter,
    task_land,
    write_validated,
)
from smoke_gemini import ExampleHeading, SmokeResult, assert_example_heading


ROOT = Path(__file__).resolve().parents[1]


class PinAndStackTests(unittest.TestCase):
    def test_imported_browser_use_is_pinned(self) -> None:
        self.assertEqual(installed_browser_use_version(), PINNED_BROWSER_USE)
        import browser_use
        from importlib.metadata import version

        self.assertEqual(version("browser-use"), "0.13.10")
        attribute = getattr(browser_use, "__version__", None)
        if attribute is not None:
            self.assertEqual(attribute, "0.13.10")

    def test_dependency_files_do_not_use_forbidden_drivers(self) -> None:
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "pyproject.toml", ROOT / "requirements.txt")
        )
        lowered = combined.lower()
        self.assertIn("browser-use==0.13.10", combined)
        for forbidden in ("playwright", "chatbrowseruse", "browser_use_api_key", "openrouter", "gemini_api_key"):
            self.assertNotIn(forbidden, lowered)

    def test_python_sources_do_not_call_forbidden_stacks(self) -> None:
        sources = [ROOT / "agent_runner.py", ROOT / "smoke_gemini.py"]
        blob = "\n".join(path.read_text(encoding="utf-8") for path in sources)
        self.assertNotIn("ChatBrowserUse", blob)
        self.assertNotIn("import playwright", blob)
        self.assertNotIn("pip install playwright", blob)
        self.assertNotIn("ChatOpenRouter", blob)
        self.assertIn("ChatGoogle", blob)
        self.assertIn("model=MODEL_NAME", blob)
        self.assertIn('MODEL_NAME = "gemini-3.8-flash"', blob)
        self.assertNotIn("gemini-2.5-flash", blob)
        self.assertIn("chromium_sandbox=False", blob)

    def test_readme_states_contabo_steps_and_proof_rule(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("python3.12", readme)
        self.assertIn("uv venv --python 3.12", readme)
        self.assertIn("uv pip install 'browser-use==0.13.10' python-dotenv", readme)
        self.assertIn("uvx browser-use install", readme)
        self.assertIn("GOOGLE_API_KEY", readme)
        self.assertIn("is_successful alone is not proof", readme)
        self.assertIn("/var/lib/adlib-agent/chrome-profile", readme)
        self.assertIn("/var/lib/adlib-agent/logs", readme)
        self.assertIn("/var/lib/adlib-agent/downloads", readme)
        self.assertIn("smoke_gemini.py", readme)
        self.assertIn("0.13.10", readme)
        self.assertIn("gemini-3.8-flash", readme)
        self.assertIn("chromium_sandbox=False", readme)
        self.assertIn("artifacts/smoke_result.json", readme)
        self.assertIn("non-empty", readme)


class SchemaTests(unittest.TestCase):
    def test_smoke_schema_rejects_empty_h1(self) -> None:
        with self.assertRaises(ValidationError):
            ExampleHeading(h1="", source_url="https://example.com")
        with self.assertRaises(ValidationError):
            SmokeResult(
                h1="",
                source_url="https://example.com",
                conversation_log_path="/tmp/conversation.txt",
                browser_use_version="0.13.10",
            )

    def test_write_validated_round_trip(self) -> None:
        result = SmokeResult(
            h1="Page heading",
            source_url="https://example.com/",
            conversation_log_path="/var/lib/adlib-agent/logs/run/conversation_1.txt",
            browser_use_version="0.13.10",
            is_successful=False,
            verified=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smoke_result.json"
            write_validated(path, result)
            loaded = SmokeResult.model_validate_json(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded.h1, "Page heading")
        self.assertTrue(loaded.h1.strip())
        self.assertTrue(loaded.conversation_log_path)
        self.assertTrue(loaded.verified)
        self.assertFalse(loaded.is_successful)

    def test_example_heading_requires_non_empty_h1(self) -> None:
        assert_example_heading(ExampleHeading(h1="Example Domain", source_url="https://example.com/"))
        assert_example_heading(ExampleHeading(h1="Document title", source_url="https://example.com/"))
        with self.assertRaises(RetryableStepError):
            assert_example_heading(ExampleHeading(h1="   ", source_url="https://example.com/"))
        with self.assertRaises(RetryableStepError):
            assert_example_heading(ExampleHeading(h1="Document title", source_url="https://other.test/"))

    def test_placeholder_extract_requires_a_card(self) -> None:
        with self.assertRaises(ValidationError):
            PlaceholderExtract(query="example", cards=[])
        parsed = PlaceholderExtract(
            query="example",
            cards=[PlaceholderCard(id="page", title="Example Domain", url="https://example.com/")],
        )
        assert_placeholder_extract(parsed, url="https://example.com", query="example", stage="extract")
        assert_placeholder_extract(
            parsed.model_copy(update={"cards": [PlaceholderCard(id="x", title="Document title", url="https://example.com/")]}),
            url="https://example.com",
            query="example",
            stage="extract",
        )
        with self.assertRaises(RetryableStepError):
            assert_placeholder_extract(
                parsed.model_copy(update={"cards": [PlaceholderCard(id="x", title="Document title", url="https://other.test/")]}),
                url="https://example.com",
                query="example",
                stage="extract",
            )

    def test_run_result_records_log_path(self) -> None:
        result = RunResult(
            run_id="20261006T000000Z-abcd1234",
            query="example",
            cards=[PlaceholderCard(id="page", title="Example Domain", url="https://example.com/")],
            conversation_log_path="/var/lib/adlib-agent/logs/run/conversation_1.txt",
            browser_use_version="0.13.10",
            verified=True,
        )
        self.assertIn("/var/lib/adlib-agent/logs/", result.conversation_log_path)

    def test_default_runtime_root(self) -> None:
        previous = os.environ.pop("ADLIB_AGENT_ROOT", None)
        try:
            paths = runtime_paths()
        finally:
            if previous is not None:
                os.environ["ADLIB_AGENT_ROOT"] = previous
        self.assertEqual(paths.root, Path("/var/lib/adlib-agent"))
        self.assertEqual(paths.chrome_profile, Path("/var/lib/adlib-agent/chrome-profile"))
        self.assertEqual(paths.logs, Path("/var/lib/adlib-agent/logs"))
        self.assertEqual(paths.downloads, Path("/var/lib/adlib-agent/downloads"))

    def test_persistent_profile_is_not_left_in_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "adlib-agent"
            paths = runtime_paths(root)
            paths.chrome_profile.mkdir(parents=True)
            paths.downloads.mkdir()
            browser = make_browser(paths)
            self.assertEqual(Path(browser.browser_profile.user_data_dir), paths.chrome_profile.resolve())
            self.assertTrue(browser.browser_profile.keep_alive)
            self.assertTrue(browser.browser_profile.headless)
            self.assertFalse(browser.browser_profile.chromium_sandbox)


class _History:
    def __init__(self, payload: str | None, errors: list[str | None] | None = None, urls: list[str] | None = None):
        self._payload = payload
        self._errors = errors or []
        self._urls = urls or []

    def errors(self) -> list[str | None]:
        return self._errors

    def urls(self) -> list[str]:
        return self._urls

    def final_result(self) -> str | None:
        return self._payload

    def get_structured_output(self, model: type[BaseModel]) -> BaseModel | None:
        if self._payload is None:
            return None
        return model.model_validate_json(self._payload)


class RetryPolicyTests(unittest.TestCase):
    def test_empty_schema_is_retryable_and_captcha_is_not(self) -> None:
        with self.assertRaises(RetryableStepError):
            parse_extract(_History(None), ExampleHeading)
        with self.assertRaises(BlockedError):
            parse_extract(_History(None, errors=["CAPTCHA challenge"]), ExampleHeading)
        with self.assertRaises(BlockedError):
            parse_extract(_History("please sign in to continue"), ExampleHeading)

    def test_429_retries_twice_then_raises(self) -> None:
        calls = {"n": 0}
        delays: list[float] = []

        class RateLimit(Exception):
            status_code = 429

        async def run_once(attempt: int) -> str:
            del attempt
            calls["n"] += 1
            raise RateLimit("429 resource exhausted")

        async def sleep(seconds: float) -> None:
            delays.append(seconds)

        with self.assertRaises(RetryableStepError):
            asyncio.run(run_step_with_retries(run_once, sleep=sleep))
        self.assertEqual(calls["n"], 3)
        self.assertEqual(delays, [5, 15])

    def test_login_wall_does_not_retry(self) -> None:
        calls = {"n": 0}

        async def run_once(attempt: int) -> str:
            del attempt
            calls["n"] += 1
            raise BlockedError("login wall")

        async def sleep(seconds: float) -> None:
            raise AssertionError(f"slept {seconds}")

        with self.assertRaises(BlockedError):
            asyncio.run(run_step_with_retries(run_once, sleep=sleep))
        self.assertEqual(calls["n"], 1)

    def test_status_classifier(self) -> None:
        class ServerError(Exception):
            status_code = 503

        with self.assertRaises(RetryableStepError):
            classify_exception(ServerError("upstream"))
        with self.assertRaises(BlockedError):
            classify_exception(RuntimeError("hcaptcha blocked the session"))
        with self.assertRaises(RuntimeError):
            classify_exception(RuntimeError("disk full"))

    def test_backoff_table_matches_harness(self) -> None:
        self.assertEqual(RETRY_BACKOFF_SECONDS, (5, 15, 45))

    def test_tasks_are_short_chain(self) -> None:
        land = task_land("https://example.com")
        filt = task_filter("https://example.com", "example")
        extract = task_extract("https://example.com", "example")
        self.assertIn("Open https://example.com", land)
        self.assertIn("2. Apply one filter", filt)
        self.assertIn("2. Extract the visible h1", extract)
        self.assertNotIn("research ads", land.lower())

    def test_second_worker_is_refused(self) -> None:
        async def overlap() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                lock = Path(tmp) / "worker.lock"
                async with single_worker(lock):
                    with self.assertRaises(WorkerBusy) as caught:
                        async with single_worker(lock):
                            pass
                    self.assertIn("concurrency is 1", str(caught.exception))

        asyncio.run(overlap())


if __name__ == "__main__":
    unittest.main()
