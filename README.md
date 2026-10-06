# Self-healing code + test generator

Software Testing mid-term project.

You give it a programming problem and a few properties the function should satisfy. It writes the code, writes tests for those properties, runs them and tells you which properties hold and which don't. In advanced mode, if a property fails, the failing input goes back to the code-writing agent so it can fix the code.

We went with option 2 from the brief (the tests have to check user-specified properties). Everything is in `pipeline.py`. We didn't use LangGraph or any other agent framework, the agents are just functions that call Gemini and pass strings to each other.

## How it works

```
problem -> Developer -> code ---------+
                                      v
properties -> QA engineer -> tests -> Judge -> report
                                      |
       advanced mode, max 3 times:    |
       failing input back to the  <---+
       Developer, or a wrong test
       back to QA
```

There are three agents:

- **Developer** writes the function from the problem text. When it's fixing, it also gets its previous code and what failed.
- **QA engineer** writes a pytest file with exactly one test per property, named `test_p1_...`, `test_p2_...` so we can match each result back to its property. It never sees the code. If it can see the code it tends to copy whatever the code does into the expected values, and then the tests don't really check anything.
- **Judge** runs the tests with pytest in a subprocess (90s timeout) and collects the result of every test case. We put a `conftest.py` in the run folder that records each test's input, expected and actual value, and the Hypothesis falsifying example. At the end the Judge also asks Gemini to write a short verdict starting with `VERDICT: PASS` or `VERDICT: FAIL`. The actual PASS/FAIL always comes from the test results, so if the LLM says something different, the results win and the report mentions it.

Problems come from HumanEval (164 problems, `openai/openai_humaneval` on Hugging Face, downloaded on the first run). The agents only get the problem text. We don't use the dataset's tests at all, and the reference solution (`canonical_solution`) is only used to check whether our generated tests are correct (see below). The agents never see it.

## Properties

You can pass properties with `--property` (repeat it for each one), or leave it out and the script will ask you to type them:

```
----- HumanEval/3 -----
def below_zero(operations: List[int]) -> bool:
    ...
Enter properties, one per line. Empty line to finish (or right away to let the LLM write them):
  > Returns True if the running balance ever goes below zero
  > Returns False for an empty list
  >
```

If you just press Enter, Gemini writes 3 to 5 properties itself. We tell it to stick to what the task and its examples actually say and to include at least one of the examples as an exact input/output, because otherwise it made up rules the task never mentions (absolute values, empty-list behaviour and so on). Properties you type in are never changed.

After a run each property ends up as one of these:

- `HOLDS`: all its test cases passed
- `VIOLATED`: at least one case failed. The report shows the input, expected and actual value.
- `NOT TESTED`: QA didn't write a test for it, so the tests get regenerated
- `BAD TEST`: the test is wrong or crashed on its own, so it doesn't say anything about the code
- `UNRESOLVED`: QA tried to fix the test twice and it still fails on the reference solution. This usually means the property is about something the task doesn't define. We stop retrying it and it doesn't count towards the verdict.

A failure in the report looks like this:

```
P2 VIOLATED (9 generated test cases, found a failing one): never negative -> False
    [FAIL] test_p2_never_negative
            input:    ops=[0]
            expected: False
            actual:   True
```

## Catching wrong tests

The LLM sometimes gets the expected value wrong, and then the Developer gets blamed for code that was actually fine. So for HumanEval problems we run the same test file a second time on the reference solution:

- fails on our code, passes on the reference: the code has a bug, send it to the Developer
- fails on the reference: the test is wrong, send it to QA

Hypothesis runs with `derandomize=True` so both runs get exactly the same inputs.

Just telling QA "this test is wrong" didn't work well. In early runs it kept writing the same wrong test. Now the conftest records the exact call the test made and what the reference returned, and QA gets that as ground truth:

```
[BAD] test_p4_negative_integers
        the reference solution fails this too, so the test is wrong
        correct:  order_by_points([-1, -2]) returns [-2, -1]
        input:    nums=[-1, -2]
        expected: [1, 2]
        actual:   [2, 1]
```

QA only gets the output for that one input, never the reference code.

Custom `--prompt` problems don't have a reference. There the Developer can push back: if it checks the failing input by hand and thinks the test is wrong, it replies `TEST_WRONG: <reason>` and the test goes back to QA. Tests that crash by themselves also go straight to QA.

## Basic vs advanced mode

Both modes test the same properties. The difference is the inputs.

**basic**: QA picks 3 example inputs per property (just 1 if the property is about one specific input) and writes them with `@pytest.mark.parametrize`. The tests run once and that's it. So 4 properties usually means about 12 test cases.

**advanced**: QA writes a Hypothesis test (`@given`) for each property, which tries up to 100 generated inputs looking for one that breaks it. When something fails we check the test against the reference first. Wrong tests go back to QA, real failures go to the Developer with the failing input, and everything runs again.

Limits in advanced mode (constants at the top of `pipeline.py`):

- Developer gets at most 3 rewrites (`MAX_HEAL`)
- QA gets at most 2 fixes per property (`MAX_TEST_FIX`), after that the property is UNRESOLVED. This is counted separately so bad tests don't use up the Developer's rewrites.
- if the test file itself is broken (syntax/import error, or a property has no test), QA regenerates the whole file once

The temperature also goes up on every retry of the same agent (0.2, 0.5, 0.8, 1.0). At a fixed low temperature it kept giving back almost the same wrong answer.

## Setup

Python 3.10+ and a free API key from Google AI Studio.

```
pip install -r requirements.txt
```

Make a `.env` file next to `pipeline.py`:

```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=model name from AI Studio
TEMPERATURE=0.2
TOP_P=0.95
MAX_OUTPUT_TOKENS=16384
```

`.env` is gitignored. You need internet the first time so it can download HumanEval.

## Running it

```
# one HumanEval problem, you'll be asked for properties
python pipeline.py --mode basic --task 0
python pipeline.py --mode advanced --task 0

# properties on the command line
python pipeline.py --mode advanced --task 3 --property "Returns True if the running balance ever goes below zero" --property "Returns False for an empty list"

# first N problems (default 5), asks for properties before each one
python pipeline.py --mode advanced --n 10

# your own problem, --name is the function name the tests will import
python pipeline.py --mode advanced --prompt "Return the index of x in list xs, or -1 if it is not there" --name find --property "If x is in xs, xs[result] == x" --property "If x is not in xs, the result is -1"
```

`--property` only works with a single problem (`--task` or `--prompt`), since each problem needs its own properties. Run `python pipeline.py -h` for all the flags.

If you want the LLM to write all the properties without being asked, pipe something in so it doesn't wait for input:

```
"" | python pipeline.py --mode advanced --n 5          # PowerShell
python pipeline.py --mode advanced --n 5 < /dev/null   # bash
```

Keep `--n` small on the free tier. Every problem is at least 3 API calls (Developer, QA, Judge), plus one if the LLM writes the properties and one for every fix. It waits and retries when it gets rate limited.

## Output

Each problem gets its own folder, e.g. `runs/advanced/HumanEval_0/`:

- `solution_v1.py`, `solution_v2.py`, ... every version of the code, `solution.py` is the latest
- `test_solution_v1.py`, ... every version of the tests, `test_solution.py` is the latest
- `result_1.txt`, `result_2.txt`, ... raw pytest output of each run
- `report.txt` the readable report: properties, every run with every test case, and a summary of all failures split into code bugs, wrong tests and broken tests
- `report.json` same thing as JSON
- `prompts.jsonl` every prompt sent to Gemini with the settings used and the reply
- `reference/` the reference solution with a copy of the tests

Basic and advanced go into separate folders so you can compare them on the same problem. Running the same problem in the same mode again overwrites its folder. Each mode folder also has a `summary.json`.

To rerun the saved tests yourself:

```
cd runs/advanced/HumanEval_0
python -m pytest test_solution.py -v
```

## Known issues

- A FAIL doesn't always mean the code is wrong. For HumanEval the reference check catches most wrong tests, but for custom prompts we only have the Developer's pushback, which can also be wrong. Check `test_solution.py` and `result_*.txt` if something looks off.
- The generated code actually runs on your machine. There's a timeout but no sandbox, so look at the code first if you use your own prompts.
- Results change between runs because the model isn't deterministic.
- Every code or test file from the LLM has to compile before we save it. If it doesn't (or the reply got cut off), we ask again up to 2 more times and keep the previous version if it still fails.
- Bigger Gemini models count their "thinking" towards `MAX_OUTPUT_TOKENS`, so if it's set too low the replies get cut off. The script prints a warning when that happens. 16384 has been fine for us.
