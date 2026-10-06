"""Open https://example.com, extract the h1, and write schema-valid JSON.

Connie smoke, plus disk proof. history.is_successful() is stored and is not
proof. Proof is artifacts/smoke_result.json validating with a non-empty h1
and a conversation log path.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent_runner import (
    MODEL_NAME,
    BlockedError,
    RetryableStepError,
    conversation_log_file,
    installed_browser_use_version,
    new_run_id,
    parse_extract,
    require_google_api_key,
    run_step_with_retries,
    write_validated,
)

SMOKE_URL = "https://example.com"
SMOKE_H1 = "Example Domain"
RESULT_PATH = Path("artifacts/smoke_result.json")
LOG_ROOT = Path("artifacts/logs/smoke")
DOWNLOADS = Path("artifacts/downloads")
SMOKE_MAX_STEPS = 8


class ExampleHeading(BaseModel):
    h1: str = Field(min_length=1)
    source_url: str = Field(min_length=1)


class SmokeResult(BaseModel):
    h1: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    conversation_log_path: str = Field(min_length=1)
    browser_use_version: str = Field(min_length=1)
    is_successful: bool | None = None
    verified: Literal[True] = True


def smoke_task() -> str:
    return (
        f"1. Open {SMOKE_URL}.\n"
        "2. Wait until the h1 is visible.\n"
        "3. Call done with h1 set to the exact heading text and source_url set to the page URL.\n"
        "4. Call done only when that JSON validates. "
        "If a CAPTCHA or login wall is on screen, stop and report it."
    )


def assert_example_heading(parsed: ExampleHeading) -> None:
    if parsed.h1.strip() != SMOKE_H1:
        raise RetryableStepError(f"expected h1 {SMOKE_H1!r}, got {parsed.h1!r}")
    if "example.com" not in parsed.source_url:
        raise RetryableStepError(f"expected an example.com URL, got {parsed.source_url!r}")


async def run_smoke() -> Path:
    require_google_api_key()
    pinned = installed_browser_use_version()
    from browser_use import Agent, Browser, ChatGoogle

    log_dir = LOG_ROOT / new_run_id()
    log_dir.mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)

    browser = Browser(headless=True, keep_alive=True, downloads_path=str(DOWNLOADS))
    llm = ChatGoogle(model=MODEL_NAME)
    agent = None
    history = None
    parsed: ExampleHeading | None = None
    task = smoke_task()
    try:
        await browser.start()
        async def on_retry(exc: RetryableStepError) -> None:
            nonlocal agent, browser
            if not exc.restart_browser:
                return
            try:
                await browser.kill()
            except Exception as kill_error:
                raise RetryableStepError(
                    f"refusing to start a second Chromium because kill failed: {kill_error}"
                ) from kill_error
            browser = Browser(headless=True, keep_alive=True, downloads_path=str(DOWNLOADS))
            await browser.start()
            agent = None

        async def run_once(_attempt: int) -> ExampleHeading:
            nonlocal agent, history
            if agent is None:
                agent = Agent(
                    task=task,
                    llm=llm,
                    browser=browser,
                    max_failures=5,
                    final_response_after_failure=True,
                    max_actions_per_step=4,
                    use_vision="auto",
                    flash_mode=False,
                    max_history_items=30,
                    output_model_schema=ExampleHeading,
                    save_conversation_path=str(log_dir),
                    calculate_cost=True,
                )
            else:
                agent.add_new_task(task)
            history = await agent.run(max_steps=SMOKE_MAX_STEPS)
            if not history.is_done():
                raise RetryableStepError(f"smoke hit max_steps={SMOKE_MAX_STEPS} without a verified h1")
            extracted = parse_extract(history, ExampleHeading)
            assert_example_heading(extracted)
            return extracted

        parsed = await run_step_with_retries(run_once, on_retry=on_retry)
        assert parsed is not None and history is not None
        log_file = conversation_log_file(log_dir)
        result = SmokeResult(
            h1=parsed.h1.strip(),
            source_url=parsed.source_url,
            conversation_log_path=str(log_file.resolve()),
            browser_use_version=pinned,
            is_successful=history.is_successful(),
            verified=True,
        )
        write_validated(RESULT_PATH, result)
        return RESULT_PATH.resolve()
    finally:
        try:
            await browser.kill()
        except Exception:
            pass


def main() -> None:
    try:
        destination = asyncio.run(run_smoke())
    except BlockedError as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(destination)


if __name__ == "__main__":
    main()
