# Self-healing code + test generator

Mid-term project for Software Testing. It's one Python script (`pipeline.py`) with three agents that pass strings to each other. No LangGraph or other agent framework.

The idea: give it a problem statement, get back Python code, tests for that code, and a pass/fail verdict. In advanced mode, if the tests find a bug, the failing input is sent back to the code-writing agent so it can fix it.

## How it works

```
problem -> Developer -> code -> QA Engineer -> tests -> Judge -> verdict
               ^                                          |
               +---- failing example (advanced mode) <----+
                     max 3 times
```

- **Developer** asks Gemini to write the function. When it's retrying, it also gets its old code and the failure output.
- **QA Engineer** asks Gemini to write a pytest file for that code.
- **Judge** is plain Python, no LLM. It saves the files, runs pytest in a subprocess (90s timeout), and reads the result. It also pulls out the Hypothesis "Falsifying example" and the coverage number.

Problems come from HumanEval (164 tasks, `openai/openai_humaneval` on Hugging Face, downloaded automatically). Only the problem text is used. The reference solutions and tests in the dataset are ignored.

## Modes

**basic**: the QA agent writes normal pytest asserts trying to hit every statement. The Judge runs them once with pytest-cov and reports pass/fail plus coverage %. No fixing loop.

**advanced**: the QA agent writes Hypothesis tests (`@given`), which try lots of generated inputs (100 per test) looking for one that breaks a property. If one does, the Judge sends that input and the traceback to the Developer, which rewrites the code. The same tests run again. This happens up to 3 times. If the test file itself is broken (import or syntax error), the tests get regenerated once instead.

## Setup

You need Python 3.10+ and a free API key from Google AI Studio.

```
pip install -r requirements.txt
copy .env.example .env
```

Then edit `.env`:

```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=model name from AI Studio
TEMPERATURE=0.2
TOP_P=0.95
MAX_OUTPUT_TOKENS=4096
```

`.env` is in `.gitignore` so the key doesn't get committed. First run needs internet to fetch HumanEval.

## Running it

```
python pipeline.py --mode basic --task 0
python pipeline.py --mode advanced --task 0
python pipeline.py --mode advanced --n 10
python pipeline.py --mode advanced --prompt "Write a function that returns the n-th Fibonacci number" --name fib
```

- `--mode`: `basic` or `advanced` (default basic)
- `--task`: run one HumanEval problem by index (0 to 163)
- `--n`: run the first n problems (default 5), used if `--task` isn't given
- `--prompt`: your own problem statement instead of HumanEval
- `--name`: function name for a custom prompt (needed so the tests know what to import)

Keep `--n` small. The free tier has rate limits and every task is at least 2 API calls. The script retries with a delay if it gets rate limited.

## Where things end up

Each task gets a folder, e.g. `runs/advanced/HumanEval_0/`:

- `solution_v1.py`, `solution_v2.py`, ...: every version of the code
- `solution.py`: the latest version (what the tests actually ran on)
- `test_solution.py`: the generated tests
- `result_v1.txt`, ...: raw pytest / coverage / Hypothesis output per attempt
- `prompts.jsonl`: every prompt sent to Gemini with the model, temperature and other settings, plus the response
- `report.json`: verdict, attempts, coverage

Each mode folder also has a `summary.json` for the whole run.

To rerun a saved result by hand:

```
cd runs/advanced/HumanEval_0
python -m pytest test_solution.py -q
```

## Terms

- **statement coverage**: % of lines in the code that ran during the tests (from pytest-cov)
- **property-based testing**: checking rules that should hold for every input (like "sorting twice = sorting once") instead of a few hand-picked examples
- **falsifying example**: the smallest input Hypothesis found that breaks a property
- **self-healing**: sending the failure back to the Developer so it fixes its own code

## Things to know

- A FAIL doesn't always mean the code is wrong. The QA agent sometimes writes a wrong expected value (HumanEval/0 did this for a threshold of 0.0). Check `test_solution.py` and `result_v*.txt` before blaming the code.
- The generated code really runs on your machine in a subprocess. There's a timeout but it isn't a sandbox, so skim the code first if you use custom prompts.
- Results change from run to run since the model isn't deterministic.
