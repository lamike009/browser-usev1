# browser-use V1 (Contabo + Gemini Flash)

One headless Chromium worker for a short browse chain. The pin is `browser-use==0.13.10`. The model is `ChatGoogle(model="gemini-2.5-flash")`. The only secret is `GOOGLE_API_KEY`.

Chromium comes from `uvx browser-use install` (CDP / browser-harness). This checkout does not use Playwright as the browser driver, Browser Use Cloud, or OpenRouter.

## Locked stack

| Item | Value |
| --- | --- |
| Package | `browser-use==0.13.10` |
| LLM | `ChatGoogle(model="gemini-2.5-flash")` |
| Env | `GOOGLE_API_KEY` |
| Browser | Chromium via `uvx browser-use install` |
| Concurrency | 1 worker |

V1 reads `GOOGLE_API_KEY` only. Do not set `GEMINI_API_KEY`, `BROWSER_USE_API_KEY`, or `OPENROUTER_API_KEY`. Do not call `ChatBrowserUse`. Do not `pip install playwright` as the driver.

## Contabo Ubuntu install

Host: Contabo Cloud VPS, Ubuntu 22.04 or 24.04, SSH as the deploy user. Outbound HTTPS must reach `generativelanguage.googleapis.com`, PyPI, and the sites the agent opens. No inbound browser port is required.

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv curl ca-certificates git
```

If `python3.12` is missing, install it from the image's packages or deadsnakes. browser-use requires Python >= 3.11 and < 4. Prefer 3.12.

Chromium system libraries: start with `uvx browser-use install` only. If Chromium fails to launch, install the shared libraries named in that error. Do not add a guessed `apt` list.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# ensure ~/.local/bin (or the uv install path) is on PATH
# source "$HOME/.local/bin/env"

git clone https://github.com/lamike009/browser-usev1.git ~/browser-use-v1
cd ~/browser-use-v1
uv venv --python 3.12
source .venv/bin/activate
uv pip install 'browser-use==0.13.10' python-dotenv
# equivalent pin file: uv pip install -r requirements.txt
uvx browser-use install
```

That last command downloads Chromium for CDP / browser-harness. It is not a Playwright install.

```bash
cp .env.example .env
chmod 600 .env
# put the Google AI Studio key in GOOGLE_API_KEY=
```

Create the runtime directories (mode `700` on the profile; never commit them):

```bash
sudo mkdir -p /var/lib/adlib-agent/{chrome-profile,logs,downloads}
sudo chown -R "$USER":"$USER" /var/lib/adlib-agent
chmod 700 /var/lib/adlib-agent /var/lib/adlib-agent/chrome-profile
```

| Path | Role |
| --- | --- |
| `/var/lib/adlib-agent/chrome-profile` | persistent Chromium `user_data_dir` |
| `/var/lib/adlib-agent/logs` | per-run conversation logs and `result.json` |
| `/var/lib/adlib-agent/downloads` | browser downloads |

`ADLIB_AGENT_ROOT` overrides that root for a local dry run. Contabo uses the default `/var/lib/adlib-agent`.

## Verify

is_successful alone is not proof. `is_done` means the agent emitted a terminal done action. `is_successful` is the agent's self-report. A run passes only when the JSON file on disk validates against the Pydantic schema and the required fields are non-empty.

```bash
cd ~/browser-use-v1 && source .venv/bin/activate

python -c "import browser_use; print(browser_use.__version__)"
# AttributeError: module 'browser_use' has no attribute '__version__'
# The 0.13.10 wheel does not set that attribute. Import the distribution version instead:

python -c "import browser_use, importlib.metadata as m; print(m.version('browser-use'))"
# 0.13.10

python agent_runner.py --check-pin
# 0.13.10

curl -sI https://generativelanguage.googleapis.com | head -n1

uvx browser-use install

python smoke_gemini.py
```

Smoke opens `https://example.com`, extracts the h1, and writes `artifacts/smoke_result.json`. The file includes `h1`, `source_url`, and `conversation_log_path`. `verified` is true only after the schema check. `is_successful` is recorded in that file and is not the pass condition.

The chained runner uses the same rule. It writes `/var/lib/adlib-agent/logs/<run_id>/result.json` plus the conversation log under that run id.

```bash
python agent_runner.py
# placeholder chain on https://example.com: land, then one filter, then extract
python agent_runner.py --url https://example.com --query example
```

Local checks that do not need Chromium or an API key:

```bash
python -m unittest tests.test_harness
```

## Harness

`agent_runner.py` keeps a single browser. browser-use 0.13.10 copies a `user_data_dir` whose path contains `chrome` into `/tmp`. The runner sets that path back to `/var/lib/adlib-agent/chrome-profile` after construction so the profile stays on disk.

```python
Browser(
    keep_alive=True,
    headless=True,
    user_data_dir="/var/lib/adlib-agent/chrome-profile",
    downloads_path="/var/lib/adlib-agent/downloads",
)
```

Tasks are short and chained with `add_new_task` on that same session:

| Step | Role | max_steps |
| --- | --- | --- |
| land | open the URL and wait for the h1 | 8 |
| filter | one placeholder text check (stands in for a search box) | 10 |
| extract | one card (`id`, `title`, `url`) through `output_model_schema` | 8 |

Agent knobs: `max_failures=5`, `final_response_after_failure=True`, `max_actions_per_step=4`, `use_vision="auto"`, `flash_mode=False`, `max_history_items=30`, `output_model_schema`, `save_conversation_path` under `/var/lib/adlib-agent/logs/<run_id>/`. Smoke sets `calculate_cost=True`. The chain leaves cost tracking off unless you pass `--calculate-cost`. There is no `fallback_llm`.

The wrapper retries a template step up to 2 times on Gemini 429 / 5xx, an empty extract, or a schema failure. Those two retries wait 5s, then 15s. The harness backoff table is `(5, 15, 45)`; 45s is the next gap and is unused while the cap stays at two retries. It does not retry a CAPTCHA or a login wall. A second process that finds `logs/worker.lock` held exits instead of starting another Chromium. On a dead browser, the runner kills that process before it starts a replacement.

The extract schema is a placeholder card, not a Meta or Google ad-library scraper. Swap the task text and the Pydantic model when a real library is in scope.

## systemd oneshot

`deploy/adlib-agent.service` is a oneshot unit. Replace `deploy` and `/home/deploy/browser-use-v1` with the deploy user and checkout path. `EnvironmentFile` points at `.env`. Do not schedule a second unit at the same time; the lock refuses the overlap, and the unit does not set `Restart`.

```bash
sudo cp deploy/adlib-agent.service /etc/systemd/system/adlib-agent.service
sudo systemctl daemon-reload
sudo systemctl start adlib-agent.service
```

## Out of scope

Contabo provisioning and SSH, real ad-library scrapers, a Playwright CDP sidecar, and Browser Use Cloud gateways.
