# Self-healing code + test generator

Mid-term project for Software Testing. It's one Python script (`pipeline.py`) with three agents that pass strings to each other. No LangGraph or other agent framework.

We went with requirement option 2 from the brief: the test generator has to check a **user-specified property** of the function. You give it a problem and a list of properties (e.g. "if x is in the list, return its index; if not, return -1"). It writes the code, writes tests for each property, runs them, and tells you which properties hold and which are violated. In advanced mode, if a property is violated, the failing input goes back to the code-writing agent so it can fix it.

## How it works

```
problem -> Developer -> code -> QA Engineer -> tests -> Judge -> verdict per property
                                   ^
                         user-specified properties
               ^                                          |
               +---- failing example (advanced mode) <----+
                     max 3 times

   tests that also fail on the dataset's reference solution:
   Judge -> QA Engineer fixes the test -> Judge
```

- **Developer** asks Gemini to write the function. When it's retrying, it also gets its old code and the failure output.
- **QA Engineer** gets the task description plus the numbered list of properties and asks Gemini for a pytest file with exactly one test per property, named `test_p1_...`, `test_p2_...` so results can be matched back to each property. It never sees the generated code (black-box testing). Otherwise it tends to copy whatever the code does into the expected values, and the tests stop being independent.
- **Judge** is plain Python, no LLM. It saves the files, runs pytest in a subprocess (90s timeout), and reads the result. It also records every test case result and pulls out the Hypothesis falsifying example. For HumanEval tasks it runs the same tests on the dataset's reference solution too, to tell wrong tests apart from wrong code.

Problems come from HumanEval (164 tasks, `openai/openai_humaneval` on Hugging Face, downloaded automatically). The agents only ever see the problem text. The dataset's reference solution (`canonical_solution`) is never shown to them. The Judge only uses it to check whether the generated tests are correct (see below). The dataset's own tests aren't used.

## Properties

The properties come from you. There are two ways to give them:

1. On the command line with `--property`, repeated for each one. Works with a single task (`--task` or `--prompt`).
2. Interactively. If you don't pass `--property`, the script prints each task and asks you to type properties one per line, then an empty line to finish:

```
----- HumanEval/3 -----
def below_zero(operations: List[int]) -> bool:
    ...
Enter properties, one per line. Empty line to finish (or right away to let the LLM write them):
  > Returns True if the running balance ever goes below zero
  > Returns False for an empty list
  >
```

If you just press Enter without typing anything, the LLM writes 3 to 5 properties from the task description instead. It's told to stick to what the task and its examples actually say, without assuming things like absolute values or empty-list behaviour. Every property has to agree with the examples, and at least one property has to be an example from the task as an exact input/output. Properties you type yourself are never changed. They get printed so you can see them. The report notes whether the properties came from the user or the LLM.

Each property ends up as one of:
- **HOLDS**: every test case for it passed
- **VIOLATED**: at least one case failed (the report shows the input, expected and actual value)
- **NOT TESTED**: the QA agent didn't write a test for it. This counts as a failure and the tests get regenerated.
- **BAD TEST**: the test is wrong or broken, so it says nothing about the code. Either it also fails on the dataset's reference solution, or (for custom prompts) it crashed by itself. These go straight to the QA agent to fix.
- **UNRESOLVED**: QA was asked to fix this property's test 2 times and it still fails on the reference solution. This usually means the property talks about something the task doesn't define. For example, "handles an empty list" when the task never says what an empty list should return, so the generated code returns `0.0` but the reference raises `ZeroDivisionError`. The pipeline stops spending rounds on it, and it no longer counts towards the verdict. It's listed separately in the summary as "Unresolved properties". The limit is `MAX_TEST_FIX` at the top of `pipeline.py`.

In the report each test case shows up as `[PASS]`, `[FAIL]` (the generated code is wrong, or `(code crashed)` if it raised an exception) or `[BAD]` (the test is wrong because the reference solution fails it too). Every failure shows the property, the exact input it ran on, and expected vs actual:

```
P2 VIOLATED (9 generated test cases, found a failing one): never negative -> False
    [FAIL] test_p2_never_negative
            input:    ops=[0]
            expected: False
            actual:   True
```

For Hypothesis tests the input is the falsifying example, already shrunk to the simplest case that fails. For basic mode it's the hand-picked case.

The end of `report.txt` has a summary that lists every failure across all runs, grouped into failures caused by the code, wrong tests and broken tests. Each entry has the run, the property, the test name, the input, and expected vs actual. The same lists are in `report.json` under `code_bugs_found`, `wrong_tests_found` and `broken_tests_found`.

### How many test cases?

- **basic**: one test function per property with 3 different hand-picked cases, so 4 properties usually give 12 test cases. A property about one specific input (like "the empty list returns []" or an example from the task) gets just that one case instead of being padded with unrelated inputs. Every case is counted and shown separately, e.g. `test_p1_found[args0-2]`, `test_p1_found[args1-0]`. The number is `CASES_PER_PROPERTY` at the top of `pipeline.py`.
- **advanced**: one Hypothesis test function per property, and each one runs up to 100 generated test cases. So 4 properties means 4 test functions but around 400 test cases. The report shows it like this:

```
run 1: PASS  (4/4 properties hold, 301 generated test cases)
P1 HOLDS (100 generated test cases): ...
P2 HOLDS (100 generated test cases): ...
P3 HOLDS (1 generated test cases): The function returns False for an empty list ...
P4 HOLDS (100 generated test cases): ...
```

A property can get fewer than 100 if there just aren't that many different inputs (P3 above is about the empty list, so there's only one). Cases tried while shrinking a failure aren't counted.

Basic mode reports the same way, but counts its hand-picked cases instead, e.g. `(3/4 properties hold, 11/12 test cases passed)` and `P1 HOLDS (3/3 test cases passed)`.

## Checking the tests with the reference solution

LLM-written tests are sometimes wrong. If the pipeline trusted them blindly, the Developer would keep getting blamed for correct code. So for HumanEval tasks, every time the tests run, the Judge also runs the exact same test file on the dataset's `canonical_solution`, which is known to be correct:

| on the generated code | on the reference | means | goes to |
|---|---|---|---|
| pass | pass | property holds | done |
| fail | pass | the generated code has a bug | Developer |
| any | fail | the test is wrong | QA Engineer |

Hypothesis is set to `derandomize=True`, so both runs get exactly the same generated inputs.

When a test is wrong, just telling QA "this is wrong" isn't enough. In early runs it kept writing the same wrong test, especially when the property text itself was misleading. So the Judge also records the exact call the test made to the function, and what the reference returned or raised for it. QA gets that as ground truth:

```
[BAD] test_p4_negative_integers
        the reference solution fails this too, so the test is wrong
        correct:  order_by_points([-1, -2]) returns [-2, -1]
        input:    nums=[-1, -2]
        expected: [1, 2]
        actual:   [2, 1]
```

QA is told that `correct:` lines win over a property's wording. This uses the reference as an oracle for single inputs only. QA never sees the reference code, and the Developer never sees any of this.

The reference copy lives in `runs/<mode>/<task>/reference/` so you can rerun it by hand.

Custom `--prompt` problems have no reference solution. For those, the Developer is allowed to push back instead: if it checks the failing input by hand and thinks the test is wrong, it replies `TEST_WRONG: <reason>` and the test goes to QA. A test that crashes by itself (e.g. Hypothesis complaining about misused decorators) also goes straight to QA.

## Modes

Both modes test the same user-specified properties. The difference is how the inputs are picked.

**basic**: for each property the QA agent hand-picks 3 example inputs (1 if the property is about one specific input) with expected outputs (`@pytest.mark.parametrize`). The Judge runs them once and reports each property. No fixing loop.

**advanced**: for each property the QA agent writes a Hypothesis test (`@given`), which tries lots of generated inputs (100 per property) looking for one that breaks it. If one does, the Judge first checks the test against the reference solution (see above). Wrong tests go back to QA to be fixed. Real failures go to the Developer with the failing input and expected vs actual, and the Developer rewrites the code. Then everything runs again.

There are separate limits, so fixing tests never uses up the Developer's chances:

- **Developer**: at most 3 rewrites (`MAX_HEAL`).
- **QA**: at most 2 fixes per property (`MAX_TEST_FIX`), then that property is marked UNRESOLVED.
- If the test file itself is broken (import or syntax error, or a property has no test), QA regenerates the whole file once.

**Temperature goes up on retries.** The first call of each agent uses `TEMPERATURE` (0.2). Each retry of the same agent adds 0.3, up to 1.0, so it's 0.2 → 0.5 → 0.8 → 1.0. At a fixed low temperature the model tends to give back almost the same answer it already got wrong. The temperature actually used is logged for every call in `prompts.jsonl`. The step is `RETRY_TEMP_STEP` at the top of `pipeline.py`.

Basic mode still checks the tests against the reference, so its report also marks wrong tests as `[BAD]`. It just doesn't try to fix anything.

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
MAX_OUTPUT_TOKENS=16384
```

`.env` is in `.gitignore` so the key doesn't get committed. First run needs internet to fetch HumanEval.

## Running it

### Flags

| Flag | What it does | Default |
|---|---|---|
| `--mode basic` / `--mode advanced` | basic = example inputs, run once. advanced = Hypothesis inputs + self-healing | `basic` |
| `--task N` | run one HumanEval problem by index (0 to 163) | none |
| `--n N` | run the first N HumanEval problems (used when there's no `--task` or `--prompt`) | `5` |
| `--prompt "..."` | your own problem statement instead of HumanEval | none |
| `--name fn` | function name for `--prompt` (the tests import it by this name). Required with `--prompt` | none |
| `--property "..."` | a property to test. Repeat the flag for each one. Only with `--task` or `--prompt` | asked interactively |
| `-h` / `--help` | prints all the flags | |

### HumanEval, one task

```
# you get asked for properties (type them, or press Enter to let the LLM write them)
python pipeline.py --mode basic --task 0
python pipeline.py --mode advanced --task 0

# properties given up front, no questions asked
python pipeline.py --mode advanced --task 3 --property "Returns True if the running balance ever goes below zero" --property "Returns False for an empty list"
```

### HumanEval, several tasks

```
# first 5 tasks (default), asks for properties before each one
python pipeline.py --mode advanced

# first 10 tasks
python pipeline.py --mode advanced --n 10
```

`--property` can't be used here since each task needs its own properties.

### Your own problem

```
# with properties
python pipeline.py --mode advanced --prompt "Return the index of x in list xs, or -1 if it is not there" --name find --property "If x is in xs, xs[result] == x" --property "If x is not in xs, the result is -1"

# without properties: you get asked, or press Enter for LLM-written ones
python pipeline.py --mode basic --prompt "Return True if n is a prime number" --name is_prime
```

### Let the LLM write all properties without being asked

The script only asks when it's run from a terminal. If you pipe something into it, it skips the question and the LLM writes the properties:

```
# PowerShell
"" | python pipeline.py --mode advanced --n 5

# Git Bash / macOS / Linux
python pipeline.py --mode advanced --n 5 < /dev/null
```

### Comparing the two modes

Run the same task in both modes. Results go to separate folders (`runs/basic/` and `runs/advanced/`) so nothing gets overwritten:

```
python pipeline.py --mode basic --task 2 --property "For a positive float, the result is at least 0 and less than 1"
python pipeline.py --mode advanced --task 2 --property "For a positive float, the result is at least 0 and less than 1"
```

Running the same task and mode again overwrites its folder.

### Rerunning saved tests by hand

```
cd runs/advanced/HumanEval_0
python -m pytest test_solution.py -v
```

### Errors you might see

- `GEMINI_API_KEY missing, add it to .env`: the `.env` file is missing or has no key
- `--prompt needs --name`: add `--name` with the function name
- `--property only works with a single task`: you used `--property` with `--n`. Use `--task` instead, or type properties when asked

Keep `--n` small. The free tier has rate limits and every task is at least 2 API calls (3 if the LLM writes the properties). The script retries with a delay if it gets rate limited.

## Where things end up

Each task gets a folder, e.g. `runs/advanced/HumanEval_0/`:

- `solution_v1.py`, `solution_v2.py`, ...: every version of the code
- `solution.py`: the latest version (what the tests actually ran on)
- `test_solution_v1.py`, `test_solution_v2.py`, ...: every version of the tests (a new version appears when QA fixes or regenerates them)
- `test_solution.py`: the latest tests
- `report.txt`: readable report. Lists the properties, then for each attempt says which ones HOLD or are VIOLATED, with every test case's inputs, expected and actual values, and the falsifying example for Hypothesis tests
- `result_1.txt`, `result_2.txt`, ...: raw pytest output for each run. Each run in the report says which code and test version it used, e.g. `run 2 (code v1, tests v2)`, and what happened next (`-> developer rewrote the code` or `-> developer says the test is wrong (...), sent back to QA`)
- `prompts.jsonl`: every prompt sent to Gemini with the model, temperature and other settings, plus the response
- `report.json`: verdict, attempts, status of each property, per-test results

Each mode folder also has a `summary.json` for the whole run.

## Terms

- **property**: a rule the function must follow, written by the user (like "sorting twice gives the same result as sorting once")
- **property-based testing with Hypothesis**: instead of a few hand-picked inputs, Hypothesis generates many inputs and checks the property on all of them
- **falsifying example**: the smallest input Hypothesis found that breaks a property
- **self-healing**: sending the failure back to the Developer so it fixes its own code

## Things to know

- Every code or test file an agent sends back has to be a complete, valid Python file before it's saved. If the reply was cut off (an opening ``` with no closing one) or doesn't compile, the same agent is asked again, up to 2 more times. If a fix is still broken after that, the previous working version is kept, so a broken file never replaces a good one.
- Models that "think" before answering (like the larger Gemini models) count that thinking towards `MAX_OUTPUT_TOKENS`. If it's too low, replies get cut off. The script prints a warning when that happens, and `prompts.jsonl` records a `finish_reason` for every call. 16384 is a safe value.

- A FAIL doesn't always mean the code is wrong. The QA agent sometimes writes a wrong expected value. For HumanEval tasks the reference check catches these. For custom prompts it's down to the Developer's TEST_WRONG pushback, which can be wrong, so check `test_solution.py` and `result_*.txt` yourself.
- The generated code really runs on your machine in a subprocess. There's a timeout but it isn't a sandbox, so skim the code first if you use custom prompts.
- Results change from run to run since the model isn't deterministic.
