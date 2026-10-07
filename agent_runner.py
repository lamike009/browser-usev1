"""Devo harness: one headless Chromium worker, short chained tasks, verified JSON.

The browser is Chromium launched by browser-use (CDP / browser-harness).
Proof of a run is schema-valid JSON on disk. history.is_successful() is only
the agent's self-report and is not proof.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import logging
import os
import re
import sys
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib.metadata import version as dist_version
from pathlib import Path
from typing import Literal, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

PINNED_BROWSER_USE = "0.13.10"
MODEL_NAME = "gemini-3.8-flash"

DEFAULT_ROOT = Path("/var/lib/adlib-agent")
MAX_STEP_RETRIES = 2
RETRY_BACKOFF_SECONDS = (5, 15, 45)
MAX_ACTIONS_PER_STEP = 4
MAX_HISTORY_ITEMS = 30

LAND_MAX_STEPS = 8
FILTER_MAX_STEPS = 10
EXTRACT_MAX_STEPS = 8

BLOCK_PATTERNS = (
    "captcha",
    "recaptcha",
    "hcaptcha",
    "verify you are human",
    "are you a robot",
    "login wall",
    "sign in to continue",
    "log in to continue",
    "please log in",
    "please sign in",
)
BROWSER_DEAD_PATTERNS = (
    "out of memory",
    "oom",
    "target crashed",
    "browser has been closed",
    "connection closed",
    "browser closed",
)
_STATUS_IN_TEXT = re.compile(r"\b(429|500|502|503|504)\b")

T = TypeVar("T")
log = logging.getLogger("adlib_agent")


class BlockedError(RuntimeError):
    """CAPTCHA, login wall, or a similar block. The wrapper does not retry these."""


class RetryableStepError(RuntimeError):
    """Gemini 429/5xx, empty extract, or schema failure. Retried up to two times."""

    def __init__(self, message: str, *, restart_browser: bool = False) -> None:
        super().__init__(message)
        self.restart_browser = restart_browser


class PlaceholderCard(BaseModel):
    """Stand-in for one ad-library card. Real scrapers replace this schema later."""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)


class PlaceholderExtract(BaseModel):
    query: str = Field(min_length=1)
    cards: list[PlaceholderCard] = Field(min_length=1)


class RunResult(BaseModel):
    run_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    cards: list[PlaceholderCard] = Field(min_length=1)
    conversation_log_path: str = Field(min_length=1)
    browser_use_version: str = Field(min_length=1)
    is_successful: bool | None = None
    verified: Literal[True] = True


class RuntimePaths(BaseModel):
    root: Path
    chrome_profile: Path
    logs: Path
    downloads: Path

    model_config = {"arbitrary_types_allowed": True}


def runtime_paths(root: Path | None = None) -> RuntimePaths:
    base = root or Path(os.environ.get("ADLIB_AGENT_ROOT", str(DEFAULT_ROOT)))
    return RuntimePaths(
        root=base,
        chrome_profile=base / "chrome-profile",
        logs=base / "logs",
        downloads=base / "downloads",
    )


def ensure_runtime_dirs(paths: RuntimePaths) -> None:
    for directory in (paths.root, paths.chrome_profile, paths.logs, paths.downloads):
        directory.mkdir(parents=True, exist_ok=True)
    paths.root.chmod(0o700)
    paths.chrome_profile.chmod(0o700)


def installed_browser_use_version() -> str:
    """Return the imported browser-use version. Must be the locked pin."""
    import browser_use

    attribute = getattr(browser_use, "__version__", None)
    distribution = dist_version("browser-use")
    if attribute and attribute != distribution:
        raise RuntimeError(
            f"browser_use.__version__={attribute!r} disagrees with distribution {distribution!r}"
        )
    resolved = attribute or distribution
    if resolved != PINNED_BROWSER_USE:
        raise RuntimeError(
            f"browser-use {resolved} is installed; V1 requires {PINNED_BROWSER_USE}"
        )
    return resolved


def require_google_api_key() -> None:
    load_dotenv()
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY. V1 does not read the deprecated Gemini env name.")
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is missing. Copy .env.example to .env and fill it in.")
    if os.environ.get("BROWSER_USE_API_KEY"):
        log.warning("BROWSER_USE_API_KEY is set. V1 does not use Browser Use Cloud.")


def write_validated(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.model_dump_json(indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    type(model).model_validate_json(path.read_text(encoding="utf-8"))


def conversation_log_file(log_dir: Path) -> Path:
    files = sorted(log_dir.glob("conversation_*.txt"))
    if not files:
        raise RetryableStepError(f"no conversation log written under {log_dir}")
    return files[-1]


def _blob_has_block(text: str) -> str | None:
    lowered = text.lower()
    for pattern in BLOCK_PATTERNS:
        if pattern in lowered:
            return pattern
    return None


def detect_block_reason(history: object) -> str | None:
    """Look at errors and URLs. Do not retry when a wall is already on screen."""
    chunks: list[str] = []
    errors = getattr(history, "errors", lambda: [])()
    urls = getattr(history, "urls", lambda: [])()
    for item in list(errors or []) + list(urls or []):
        if item:
            chunks.append(str(item))
    return _blob_has_block("\n".join(chunks))


def classify_exception(exc: BaseException) -> None:
    """Re-raise as BlockedError or RetryableStepError, or re-raise unknown errors."""
    if isinstance(exc, (BlockedError, RetryableStepError, SystemExit, KeyboardInterrupt)):
        raise exc
    text = f"{type(exc).__name__}: {exc}"
    blocked = _blob_has_block(text)
    if blocked:
        raise BlockedError(f"{blocked}: {exc}") from exc
    status = getattr(exc, "status_code", None)
    if status == 429 or (isinstance(status, int) and 500 <= status <= 599):
        raise RetryableStepError(str(exc)) from exc
    if _STATUS_IN_TEXT.search(text) or any(
        phrase in text.lower() for phrase in ("rate limit", "resource exhausted", "too many requests")
    ):
        raise RetryableStepError(str(exc)) from exc
    if any(phrase in text.lower() for phrase in BROWSER_DEAD_PATTERNS):
        raise RetryableStepError(str(exc), restart_browser=True) from exc
    raise exc


def parse_extract(history: object, model: type[T]) -> T:
    blocked = detect_block_reason(history)
    if blocked:
        raise BlockedError(blocked)
    getter = getattr(history, "get_structured_output", None)
    parsed = None
    if getter is not None:
        try:
            parsed = getter(model)
        except ValidationError as exc:
            final = getattr(history, "final_result", lambda: None)()
            final_block = _blob_has_block(str(final)) if final else None
            if final_block:
                raise BlockedError(final_block) from exc
            raise RetryableStepError(f"schema validation failed: {exc}") from exc
    if parsed is None:
        final = getattr(history, "final_result", lambda: None)()
        final_block = _blob_has_block(str(final)) if final else None
        if final_block:
            raise BlockedError(final_block)
        raise RetryableStepError("empty structured output")
    try:
        return model.model_validate(parsed)
    except ValidationError as exc:
        raise RetryableStepError(f"schema validation failed: {exc}") from exc


def assert_placeholder_extract(parsed: PlaceholderExtract, *, url: str, query: str, stage: str) -> None:
    if stage == "land" and parsed.query != "landing":
        raise RetryableStepError(f"land query was {parsed.query!r}, expected 'landing'")
    if stage != "land" and parsed.query != query:
        raise RetryableStepError(f"{stage} query was {parsed.query!r}, expected {query!r}")
    if not parsed.cards:
        raise RetryableStepError("extract contained no cards")
    if "example.com" in url:
        titles = [card.title.strip() for card in parsed.cards]
        urls = [card.url for card in parsed.cards]
        if not any(titles):
            raise RetryableStepError(f"expected a non-empty heading, got {titles}")
        if not any("example.com" in card_url for card_url in urls):
            raise RetryableStepError(f"expected an example.com card url, got {urls}")


async def run_step_with_retries(
    run_once: Callable[[int], Awaitable[T]],
    *,
    on_retry: Callable[[RetryableStepError], Awaitable[None]] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run one template step. Retry up to twice. Never retry a block."""
    last: RetryableStepError | None = None

    async def _backoff(exc: RetryableStepError, attempt: int) -> bool:
        nonlocal last
        last = exc
        if attempt >= MAX_STEP_RETRIES:
            return True
        if on_retry is not None:
            await on_retry(exc)
        delay = RETRY_BACKOFF_SECONDS[attempt]
        log.warning("step failed (%s); retry %s in %ss", exc, attempt + 1, delay)
        await sleep(delay)
        return False

    for attempt in range(MAX_STEP_RETRIES + 1):
        try:
            return await run_once(attempt)
        except BlockedError:
            raise
        except RetryableStepError as exc:
            if await _backoff(exc, attempt):
                break
        except Exception as exc:
            try:
                classify_exception(exc)
            except RetryableStepError as wrapped:
                if await _backoff(wrapped, attempt):
                    break
    assert last is not None
    raise last


def task_land(url: str) -> str:
    return (
        f"1. Open {url} exactly.\n"
        "2. Wait until the page has loaded.\n"
        "3. Call done with query \"landing\" and one card: "
        "id \"page\", title set to the visible h1 or, if there is no h1, the document title, "
        "url set to the final page URL.\n"
        "4. Call done only when that JSON validates. "
        "If a CAPTCHA or login wall is on screen, stop and report it. Do not guess the heading."
    )


def task_filter(url: str, query: str) -> str:
    return (
        f"1. Stay on {url}. Open it only if it is not already the current page.\n"
        f"2. Apply one filter: confirm the visible page text contains \"{query}\". "
        "This placeholder stands in for a single search box.\n"
        f"3. Call done with query \"{query}\" and one card whose title is the visible h1 "
        "(or the document title if there is no h1) and whose url is the page URL.\n"
        "4. If the text is missing, or a CAPTCHA or login wall is on screen, stop and report it. "
        "Do not invent a card and do not open another site."
    )


def task_extract(url: str, query: str) -> str:
    return (
        f"1. Stay on {url}. Open it only if it is not already the current page.\n"
        "2. Extract the visible h1, or the document title if there is no h1, and the page URL "
        "through the done action only.\n"
        f"3. The schema requires query \"{query}\" and cards with at least one item "
        "(id, title, url). title must be that non-empty heading.\n"
        "4. Call done only when the JSON validates against the schema. "
        "Otherwise report the fail reason. "
        "If a CAPTCHA or login wall is on screen, stop and report it."
    )


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


class WorkerBusy(RuntimeError):
    """A second Chromium worker tried to start while one lock is held."""


@asynccontextmanager
async def single_worker(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise WorkerBusy("another worker holds the browser lock; concurrency is 1") from exc
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def make_browser(paths: RuntimePaths):
    from browser_use import Browser

    browser = Browser(
        headless=True,
        keep_alive=True,
        user_data_dir=str(paths.chrome_profile),
        downloads_path=str(paths.downloads),
        chromium_sandbox=False,
    )
    # 0.13.10 copies a user_data_dir to /tmp when the path contains "chrome"
    # (it assumes a system Chrome profile). Point the live profile back at the
    # persistent directory so cookies survive the run.
    browser.browser_profile.user_data_dir = str(paths.chrome_profile.resolve())
    return browser


def make_agent(task: str, llm: object, browser: object, log_dir: Path, *, calculate_cost: bool):
    from browser_use import Agent

    return Agent(
        task=task,
        llm=llm,
        browser=browser,
        max_failures=5,
        final_response_after_failure=True,
        max_actions_per_step=MAX_ACTIONS_PER_STEP,
        use_vision="auto",
        flash_mode=False,
        max_history_items=MAX_HISTORY_ITEMS,
        output_model_schema=PlaceholderExtract,
        save_conversation_path=str(log_dir),
        calculate_cost=calculate_cost,
    )


async def _relaunch(browser: object, paths: RuntimePaths):
    """Kill the current Chromium, then start one replacement. Never two at once."""
    killer = getattr(browser, "kill", None)
    if killer is None:
        raise RetryableStepError("browser has no kill(); refusing to start a second Chromium")
    try:
        await killer()
    except Exception as exc:
        raise RetryableStepError(
            f"refusing to start a second Chromium because kill failed: {exc}"
        ) from exc
    fresh = make_browser(paths)
    await fresh.start()
    return fresh


async def run_chain(
    *,
    url: str = "https://example.com",
    query: str = "example",
    root: Path | None = None,
    calculate_cost: bool = False,
) -> Path:
    """Land, filter, then extract on one kept-alive browser. Return result.json."""
    require_google_api_key()
    pinned = installed_browser_use_version()
    paths = runtime_paths(root)
    ensure_runtime_dirs(paths)
    run_id = new_run_id()
    log_dir = paths.logs / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    from browser_use import ChatGoogle

    steps: list[tuple[str, str, int]] = [
        ("land", task_land(url), LAND_MAX_STEPS),
        ("filter", task_filter(url, query), FILTER_MAX_STEPS),
        ("extract", task_extract(url, query), EXTRACT_MAX_STEPS),
    ]

    async with single_worker(paths.logs / "worker.lock"):
        browser = make_browser(paths)
        llm = ChatGoogle(model=MODEL_NAME)
        agent = None
        history = None
        parsed: PlaceholderExtract | None = None
        try:
            await browser.start()
            for stage, task, max_steps in steps:
                log.info("template %s (max_steps=%s)", stage, max_steps)

                async def on_retry(exc: RetryableStepError, _stage: str = stage) -> None:
                    nonlocal agent, browser
                    if exc.restart_browser:
                        log.warning("restarting the single Chromium worker after %s", _stage)
                        browser = await _relaunch(browser, paths)
                        agent = None

                async def run_once(
                    _attempt: int,
                    _stage: str = stage,
                    _task: str = task,
                    _steps: int = max_steps,
                ) -> PlaceholderExtract:
                    nonlocal agent, history
                    if agent is None:
                        agent = make_agent(_task, llm, browser, log_dir, calculate_cost=calculate_cost)
                    else:
                        agent.add_new_task(_task)
                    history = await agent.run(max_steps=_steps)
                    if not history.is_done():
                        blocked = detect_block_reason(history)
                        if blocked:
                            raise BlockedError(blocked)
                        raise RetryableStepError(f"{_stage} hit max_steps={_steps} without a verified extract")
                    extracted = parse_extract(history, PlaceholderExtract)
                    assert_placeholder_extract(extracted, url=url, query=query, stage=_stage)
                    return extracted

                parsed = await run_step_with_retries(run_once, on_retry=on_retry)

            assert parsed is not None and history is not None
            log_file = conversation_log_file(log_dir)
            result = RunResult(
                run_id=run_id,
                query=parsed.query,
                cards=parsed.cards,
                conversation_log_path=str(log_file.resolve()),
                browser_use_version=pinned,
                is_successful=history.is_successful(),
                verified=True,
            )
            destination = log_dir / "result.json"
            write_validated(destination, result)
            log.info("verified extract written to %s", destination)
            return destination
        finally:
            try:
                await browser.kill()
            except Exception:
                log.exception("browser kill at shutdown failed")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Run one chained browser-use worker (concurrency 1).")
    parser.add_argument("--url", default="https://example.com", help="Page the placeholder chain opens.")
    parser.add_argument("--query", default="example", help="Single filter/search string.")
    parser.add_argument(
        "--calculate-cost",
        action="store_true",
        help="Ask browser-use to track Gemini token cost for this run.",
    )
    parser.add_argument(
        "--check-pin",
        action="store_true",
        help="Print the imported browser-use version and exit.",
    )
    args = parser.parse_args(argv)
    if args.check_pin:
        print(installed_browser_use_version())
        return
    try:
        destination = asyncio.run(
            run_chain(url=args.url, query=args.query, calculate_cost=args.calculate_cost)
        )
    except BlockedError as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except WorkerBusy as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(3) from exc
    print(destination)


if __name__ == "__main__":
    main()
