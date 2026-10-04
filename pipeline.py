"""Developer -> QA engineer -> Judge pipeline for user-specified property testing.

Usage:
    python pipeline.py --mode basic --task 0
    python pipeline.py --mode advanced --n 5
    python pipeline.py --mode advanced --prompt "Return the index of x in list xs, -1 if missing" \
        --name find --property "If x is in xs, xs[result] == x" --property "If x is not in xs, returns -1"
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
MAX_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "16384"))
MAX_HEAL = 3      # developer fix rounds
MAX_TEST_FIX = 2  # QA fixes per property before its test is marked unresolved, separate from MAX_HEAL
RETRY_TEMP_STEP = 0.3  # each retry of the same agent gets a bit more temperature, so it doesn't repeat itself
CASES_PER_PROPERTY = 3  # basic mode
TEST_TIMEOUT = 90
RUNS_DIR = Path("runs")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DEV_SYSTEM = (
    "You are a Python developer. Reply with one complete Python module inside a "
    "```python block. Include needed imports. No explanations, no example usage, "
    "no input() or print() calls."
)
PROP_SYSTEM = (
    "You are a QA engineer. Read the task and list 3 to 5 properties the function "
    "must satisfy, covering normal cases and edge cases (e.g. for a search function: "
    "the expected output when the element is found, and when it is not found). "
    "Rules: "
    "1) Base every property only on what the task text and its examples say. Don't "
    "assume behaviour the task doesn't state, such as using absolute values, rounding, "
    "or what happens for empty or invalid input. "
    "2) The examples in the task are the most precise part of the spec. Every property "
    "must agree with every example. If an example shows an unusual rule (e.g. how "
    "negative numbers or ties are handled), state that rule exactly as the example "
    "shows it. "
    "3) Include at least one property that states an example from the task as an "
    "exact input and output. "
    "One property per line as a plain sentence. No numbering, no code, no other text."
)
QA_BASIC_SYSTEM = (
    "You are a QA engineer. You get a task description and a numbered list of "
    "properties the function must satisfy. You don't see the implementation, so base "
    "expected values on the task only. Write a pytest file that checks each property with hand-picked "
    "example inputs. Start with `import pytest` and `from solution import {name}`. "
    "Write exactly one test per property, named test_p<number>_<short_name> in the "
    "same order as the list. Each test is a @pytest.mark.parametrize over exactly "
    f"{CASES_PER_PROPERTY} different (args, expected) cases that the property is about. If "
    "the property is about one specific input (e.g. the empty list, or an example "
    "from the task), use just that one case instead of padding with other inputs. "
    "Each test does "
    "`actual = {name}(*args)` and "
    "`assert actual == expected, f\"expected={{expected!r}} actual={{actual!r}}\"`. "
    "Test only the listed properties. Reply with one ```python block only."
)
QA_ADV_SYSTEM = (
    "You are a QA engineer. You get a task description and a numbered list of "
    "properties the function must satisfy. You don't see the implementation, so base "
    "expected values on the task only. Write a pytest file that checks each property with the hypothesis "
    "library. Start with `from hypothesis import given, settings, strategies as st` "
    "and `from solution import {name}`. Write exactly one test per "
    "property, named test_p<number>_<short_name> in the same order as the list. Use "
    "@given with strategies that only produce inputs the task allows and that hit "
    "the cases the property talks about, plus @settings(max_examples=100, "
    "deadline=None). Every assert needs a message like "
    "f\"expected={{expected!r}} actual={{actual!r}}\". Test only the listed "
    "properties. Reply with one ```python block only."
)

JUDGE_SYSTEM = (
    "You are the test executor and judge. The generated tests have already been run, "
    "and you get the task, the final code and the execution results. Write the verdict "
    "report. The first line is exactly 'VERDICT: PASS' or 'VERDICT: FAIL'. It's PASS "
    "only if every property that counts holds (unresolved properties don't count). Then "
    "write 3 to 6 short lines: which properties hold, which failed and on what input with "
    "expected vs actual, whether each failure came from the code or from a wrong test, and "
    "for code failures, what the code does wrong. Use only the results given and don't "
    "invent any. Plain text, no markdown."
)

# dropped into the run folder so pytest records one result per test case
CONFTEST = '''import json, os, re
import pytest
from hypothesis import settings

# same generated inputs on every run, so the code and the reference see the same cases
settings.register_profile("judge", derandomize=True, database=None)
settings.load_profile("judge")

results = []

# wrap the function under test so we know the last call a test made and what came back.
# on the reference run this is the ground truth for the input that broke the test
FUNC = os.environ.get("JUDGE_FUNC")
last_call = {}
try:
    import solution
    _orig = getattr(solution, FUNC)

    def _short(text, n=300):
        return text if len(text) <= n else text[:n] + "..."

    def _wrapped(*args, **kwargs):
        last_call.clear()
        shown = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        last_call["call"] = _short(f"{FUNC}({', '.join(shown)})")
        try:
            out = _orig(*args, **kwargs)
        except Exception as e:
            last_call["raised"] = _short(f"{type(e).__name__}: {e}")
            raise
        last_call["returned"] = _short(repr(out))
        return out

    setattr(solution, FUNC, _wrapped)
except Exception:
    pass  # solution.py broken or missing, pytest will report it


def pytest_runtest_setup(item):
    last_call.clear()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    rep = (yield).get_result()
    cs = getattr(item, "callspec", None)
    rep.case_params = {k: repr(v) for k, v in cs.params.items()} if cs else None
    # count the inputs hypothesis actually ran, leaving out the shrinking phase
    stats = getattr(item, "hypothesis_statistics", None)
    rep.inputs_tried = None
    if stats and call.when == "call":
        rep.inputs_tried = 0
        for phase in stats.split("- during ")[1:]:
            m = re.search(r"(\\d+) passing.*?(\\d+) failing", phase)
            if m and not phase.startswith("shrink"):
                rep.inputs_tried += int(m.group(1)) + int(m.group(2))


def pytest_runtest_logreport(report):
    if report.when != "call" and not (report.when == "setup" and report.failed):
        return
    entry = {
        "test": report.nodeid.split("::", 1)[-1],
        "outcome": report.outcome,
        "params": getattr(report, "case_params", None),
        "inputs_tried": getattr(report, "inputs_tried", None),
    }
    if report.failed:
        text = str(report.longrepr)
        crash = getattr(report.longrepr, "reprcrash", None)
        msg = crash.message if crash else text
        path = str(getattr(crash, "path", "")).replace("\\\\", "/")
        # assertion = property violated, code_error = solution.py crashed, test_error = the test itself is broken
        if msg.startswith(("AssertionError", "assert ")):
            entry["kind"] = "assertion"
        elif path.endswith("/solution.py"):
            entry["kind"] = "code_error"
        else:
            entry["kind"] = "test_error"
        entry["message"] = msg.replace("AssertionError: ", "", 1).splitlines()[0]
        m = re.search(r"(?:Falsifying example|Failing test case):.*?(?:\\n\\s*\\n|\\Z)", text, re.DOTALL)
        if m:
            example = re.sub(r"^E {1,3}", "", m.group(0), flags=re.M)
            entry["falsifying_example"] = example.split("\\nExplanation:")[0].strip()
        if last_call:
            entry["last_call"] = dict(last_call)
    results.append(entry)


def pytest_sessionfinish(session):
    with open("results.json", "w") as f:
        json.dump(results, f)
'''


def retry_temperature(n):
    """Temperature for the n-th retry of an agent (n=0 is the first try)."""
    return round(min(TEMPERATURE + RETRY_TEMP_STEP * n, 1.0), 2)


def call_gemini(system, user, log, tag, temperature=TEMPERATURE):
    """Single Gemini call; logs prompt + settings to the run's prompts.jsonl."""
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        top_p=TOP_P,
        max_output_tokens=MAX_TOKENS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    resp = None
    for attempt in range(5):
        try:
            resp = client.models.generate_content(model=MODEL, contents=user, config=cfg)
            break
        except Exception as e:
            wait = 5 * 2 ** attempt
            print(f"  gemini error ({e.__class__.__name__}), retrying in {wait}s")
            time.sleep(wait)
    if resp is None:
        raise RuntimeError("Gemini call failed after retries")
    text = resp.text or ""
    finish = str(resp.candidates[0].finish_reason) if resp.candidates else None
    if finish and "MAX_TOKENS" in finish:
        print(f"  warning: {tag} reply was cut off at MAX_OUTPUT_TOKENS={MAX_TOKENS}, raise it in .env")

    entry = {
        "agent": tag,
        "model": MODEL,
        "temperature": temperature,
        "top_p": TOP_P,
        "max_output_tokens": MAX_TOKENS,
        "system_prompt": system,
        "user_prompt": user,
        "finish_reason": finish,
        "response": text,
    }
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return text


def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if not m:
        m = re.search(r"```(?:python)?\n(.*)", text, re.DOTALL)  # opening fence but reply cut off
    return (m.group(1) if m else text).strip() + "\n"


def syntax_error(code):
    try:
        compile(code, "<generated>", "exec")
        return None
    except SyntaxError as e:
        return f"{e.msg} (line {e.lineno})"


def ask_for_code(system, user, log, tag, temperature=TEMPERATURE, first_reply=None, fallback=None):
    """Get a Python file from the LLM and refuse anything that doesn't compile.

    Asks again up to 2 times. If it still isn't valid, returns fallback (the last
    working version) when there is one, so a broken file never replaces a good one.
    """
    text, err = first_reply, None
    for _ in range(3):
        if text is None:
            note = "" if err is None else (
                f"\n\nYour last reply was not a complete, valid Python file ({err}). "
                "Reply with the complete file in a single ```python block and nothing else.")
            text = call_gemini(system, user + note, log, tag, temperature)
        code = extract_code(text)
        if text.count("```") % 2:
            err = "reply was cut off before the closing ```"  # may still compile, but it's half a file
        else:
            err = syntax_error(code)
        if err is None:
            return code
        print(f"  {tag} returned invalid Python: {err}, asking again")
        text = None
    if fallback is not None:
        print(f"  {tag} still invalid after 3 tries, keeping the previous version")
        return fallback
    print(f"  {tag} still invalid after 3 tries, using it anyway (the judge will report it)")
    return code


def developer(task, log):
    return ask_for_code(DEV_SYSTEM, f"Task:\n{task}", log, "developer")


def developer_fix(task, code, feedback, log, can_blame_test, retry):
    """Returns ("code", new_code), or ("test_wrong", reason) when pushback is allowed.

    Pushback is only allowed when there is no reference solution to check the tests with.
    """
    user = (
        f"Task:\n{task}\n\nYour previous code:\n```python\n{code}```\n\n"
        f"It failed these tests:\n{feedback}\n\n"
    )
    if can_blame_test:
        user += (
            "Check the failing input against the task by hand. If your code is wrong, fix "
            "the logic and return the full corrected module. If your code is right for "
            "that input and the test's expected value is wrong, reply with only one line: "
            "TEST_WRONG: <short reason>"
        )
    else:
        user += ("These tests pass on a known-correct implementation, so the bug is in your "
                 "code. Fix the logic and return the full corrected module.")
    temp = retry_temperature(retry)
    text = call_gemini(DEV_SYSTEM, user, log, "developer_fix", temp)
    m = re.search(r"TEST_WRONG:\s*(.+)", text)
    if can_blame_test and m and "```" not in text:
        return "test_wrong", m.group(1).strip()
    return "code", ask_for_code(DEV_SYSTEM, user, log, "developer_fix", temp, first_reply=text, fallback=code)


def generate_properties(task, log):
    """Fallback when the user gives no properties: ask the LLM to write them."""
    text = call_gemini(PROP_SYSTEM, f"Task:\n{task}", log, "property_writer")
    props = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip() for line in text.splitlines()]
    return [p for p in props if p and not p.startswith("```")]


def qa_engineer(task, name, props, mode, log):
    """Black-box: QA writes tests from the task and properties, never sees the code."""
    system = (QA_BASIC_SYSTEM if mode == "basic" else QA_ADV_SYSTEM).format(name=name)
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(props, 1))
    user = f"Task:\n{task}\n\nProperties:\n{numbered}"
    return ask_for_code(system, user, log, f"qa_{mode}")


def qa_fix(task, name, props, mode, tests, reason, feedback, log, retry):
    """Called when a test is wrong or broken. QA still doesn't see the code."""
    system = (QA_BASIC_SYSTEM if mode == "basic" else QA_ADV_SYSTEM).format(name=name)
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(props, 1))
    user = (
        f"Task:\n{task}\n\nProperties:\n{numbered}\n\n"
        f"Your test file:\n```python\n{tests}```\n\nIt failed with:\n{feedback}\n\n"
        f"Why it is flagged: {reason}\n\n"
        "Lines starting with 'correct:' show what a known-correct implementation returns "
        "(or raises) for the exact call your test made. Treat them as ground truth: your "
        "expected values must agree with them, even where a property's wording suggests "
        "otherwise. If your test computes expected values with a helper function or an "
        "inline formula, apply it to every 'correct:' input. If it doesn't give the "
        "correct output, the helper's logic is wrong: change the logic itself, not just "
        "tie-breaking or the inputs, until it matches all of them. Don't explain the "
        "difference with rules the task doesn't state. Fix the failing tests and return "
        "the full test file."
    )
    return ask_for_code(system, user, log, f"qa_fix_{mode}", retry_temperature(retry), fallback=tests)


def judge(run_dir, name):
    """Run pytest on solution.py + test_solution.py in run_dir."""
    (run_dir / "conftest.py").write_text(CONFTEST, encoding="utf-8")
    (run_dir / "results.json").unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "pytest", "test_solution.py", "-q", "--tb=short", "-p", "no:cacheprovider"]
    try:
        # no __pycache__ in the run folders, and tell conftest which function to watch
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "JUDGE_FUNC": name}
        p = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True, timeout=TEST_TIMEOUT, env=env)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        out, rc = f"Timed out after {TEST_TIMEOUT}s (possible infinite loop)", -1

    tests = []
    rfile = run_dir / "results.json"
    if rfile.exists():
        tests = json.loads(rfile.read_text())
        rfile.unlink()

    return {
        "passed": rc == 0,
        "returncode": rc,
        "tests": tests,
        "bad_tests": rc in (2, 4),  # collection/usage error, tests themselves are broken
        "output": out,
    }


def judge_verdict(task, code, facts, verdict, log):
    """Judge agent: turns the execution results into a verdict report.

    The PASS/FAIL itself comes from the test results, so if the LLM says something
    different the results win and the mismatch is noted.
    """
    user = f"Task:\n{task}\n\nFinal code:\n```python\n{code}```\n\nTest execution results:\n{facts}"
    text = call_gemini(JUDGE_SYSTEM, user, log, "judge").strip()
    said = re.match(r"VERDICT:\s*(PASS|FAIL)", text)
    if not said:
        return f"VERDICT: {verdict}\n{text}"
    if said.group(1) != verdict:
        return (f"VERDICT: {verdict}  (the judge agent said {said.group(1)}, but the test "
                f"results say {verdict}, so the results win)\n{text[said.end():].strip()}")
    return text


def describe(t):
    """A test result: status, name, the input it ran on and expected/actual."""
    label = {"passed": "PASS", "invalid": "BAD", "unresolved": "UNRESOLVED"}.get(t["outcome"], "FAIL")
    line = f"[{label}] {t['test']}"
    if t.get("kind") == "test_error":
        line += "  (error in the test itself)"
    elif t.get("kind") == "code_error":
        line += "  (code crashed)"

    inp = test_input(t)
    if t["outcome"] == "passed":
        if inp is not None:
            line += f"  input: {inp}"
            if t["params"].get("expected"):
                line += f" -> {t['params']['expected']}"
        return line

    if t.get("kind") == "bad_test":
        line += "\n        the reference solution fails this too, so the test is wrong"
        ref = t.get("reference_call")
        if ref:
            what = f"returns {ref['returned']}" if "returned" in ref else f"raises {ref.get('raised')}"
            line += f"\n        correct:  {ref['call']} {what}"
    if inp is not None:
        line += f"\n        input:    {inp}"
    exp_act = split_expected_actual(t.get("message", ""))
    if exp_act:
        line += f"\n        expected: {exp_act[0]}\n        actual:   {exp_act[1]}"
    elif t.get("message"):
        line += f"\n        error:    {t['message']}"
    return line


def test_input(t):
    """The input a test case ran on, as text. None if unknown."""
    params = t.get("params")
    if params:
        if "args" in params:
            args = params["args"]
            if args.startswith("(") and args.endswith(")"):
                args = args[1:-1].rstrip(",")
            return args
        return ", ".join(f"{k}={v}" for k, v in params.items() if k != "expected")
    ex = t.get("falsifying_example")
    if not ex or "(" not in ex:
        # plain test with no parameters: fall back to the call it made to the function
        return t["last_call"]["call"] if t.get("last_call") else None
    # "Falsifying example: test_x(\n    ops=[0],\n    w=1,  # or any other value\n)" -> "ops=[0], w=1"
    parts = []
    for ln in ex.split("(", 1)[1].splitlines():
        ln = ln.split("  #")[0].strip().rstrip(",")
        if ln.endswith(")") and ln.count(")") > ln.count("("):
            ln = ln[:-1].rstrip(",")
        if ln and not ln.startswith("#"):
            parts.append(ln)
    return ", ".join(parts)


def split_expected_actual(msg):
    m = re.search(r"expected=(.*) actual=(.*)$", msg.strip())
    return (m.group(1), m.group(2)) if m else None


def feedback_from(tests, props, output=""):
    """Failed tests with the property each one checks, for the developer or QA."""
    lines = []
    for t in tests:
        if t["outcome"] in ("passed", "unresolved"):
            continue
        m = re.match(r"test_p(\d+)", t["test"])
        if m and int(m.group(1)) <= len(props):
            lines.append(f"Property P{m.group(1)}: {props[int(m.group(1)) - 1]}")
        lines.append(describe(t))
    if lines:
        return "\n".join(lines)
    return "\n".join(output.splitlines()[-40:])


def property_status(tests, props):
    """Group test results by property using the test_p<n>_ naming."""
    groups = {i: [] for i in range(1, len(props) + 1)}
    for t in tests:
        m = re.match(r"test_p(\d+)", t["test"])
        if m and int(m.group(1)) in groups:
            groups[int(m.group(1))].append(t)
    status = []
    for i, prop in enumerate(props, 1):
        ts = groups[i]
        ok = sum(t["outcome"] == "passed" for t in ts)
        failed = [t for t in ts if t["outcome"] != "passed"]
        if not ts:
            verdict = "NOT TESTED"
        elif any(t["outcome"] == "unresolved" for t in ts):
            verdict = "UNRESOLVED"  # QA couldn't write a test that agrees with the reference
        elif not failed:
            verdict = "HOLDS"
        elif all(t.get("kind") in ("test_error", "bad_test") for t in failed):
            verdict = "BAD TEST"  # the test is broken or wrong, says nothing about the code
        else:
            verdict = "VIOLATED"
        gen = generated_cases(ts)
        status.append({"id": f"P{i}", "property": prop, "status": verdict,
                       "passed": ok, "total": len(ts),
                       "test_cases": gen if gen is not None else len(ts), "tests": ts})
    return status


def save_tests(run_dir, tests, v):
    (run_dir / f"test_solution_v{v}.py").write_text(tests, encoding="utf-8")
    (run_dir / "test_solution.py").write_text(tests, encoding="utf-8")


def generated_cases(tests):
    """Hypothesis test cases run, or None if these aren't hypothesis tests."""
    counts = [t["inputs_tried"] for t in tests if t.get("inputs_tried") is not None]
    return sum(counts) if counts else None


def render_attempt(label, res, props):
    # basic: every parametrized case is a test case
    # advanced: one hypothesis test per property, every generated input is a test case
    tests = res["tests"]
    status = property_status(tests, props)
    holds = sum(s["status"] == "HOLDS" for s in status)
    gen = generated_cases(tests)
    head = f"{label}: {'PASS' if res['passed'] else 'FAIL'}  ({holds}/{len(props)} properties hold, "
    n_unres = sum(s["status"] == "UNRESOLVED" for s in status)
    if n_unres:
        head += f"{n_unres} unresolved, "
    if gen is not None:
        head += f"{gen} generated test cases)"
    else:
        passed = sum(t["outcome"] == "passed" for t in tests)
        head += f"{passed}/{len(tests)} test cases passed)"
    if not tests:
        return head, [res["output"].strip()[-800:]]

    body = []
    for s in status:
        n = generated_cases(s["tests"])
        if n is not None:
            count = f"{n} generated test cases" + (", found a failing one" if s["status"] == "VIOLATED" else "")
        else:
            count = f"{s['passed']}/{s['total']} test cases passed"
        body.append(f"{s['id']} {s['status']} ({count}): {s['property']}")
        body += ["    " + describe(t).replace("\n", "\n    ") for t in s["tests"]]
    return head, body


def check_against_reference(res, run_dir, ref_dir, name):
    """Run the same tests on the dataset's reference solution.

    Any test that fails there is wrong, whatever the generated code did.
    """
    shutil.copy(run_dir / "test_solution.py", ref_dir / "test_solution.py")
    ref = judge(ref_dir, name)
    ref_failed = {t["test"]: t for t in ref["tests"] if t["outcome"] != "passed"}
    for t in res["tests"]:
        if t["test"] in ref_failed:
            r = ref_failed[t["test"]]
            t["outcome"] = "invalid"
            t["kind"] = "bad_test"
            t["message"] = f"fails on the reference solution too: {r.get('message', '')}"
            if r.get("last_call"):
                t["reference_call"] = r["last_call"]  # what the correct function did on that input
            if r.get("falsifying_example"):
                t["falsifying_example"] = r["falsifying_example"]
        elif t.get("kind") == "test_error":
            t["kind"] = "code_error"  # the test only breaks on our code, so the code caused it
    if ref_failed:
        res["passed"] = False
    res["reference_failures"] = len(ref_failed)


def prop_number(t):
    m = re.match(r"test_p(\d+)", t["test"])
    return int(m.group(1)) if m else None


def apply_unresolved(res, unresolved):
    """Tests of unresolved properties stop counting towards the verdict."""
    for t in res["tests"]:
        if prop_number(t) in unresolved and t["outcome"] != "passed":
            t["outcome"] = "unresolved"
    if res["tests"] and res["returncode"] in (0, 1):
        res["passed"] = all(t["outcome"] in ("passed", "unresolved") for t in res["tests"])


def run_task(task_id, task, name, props, mode, out_root, reference=None):
    run_dir = out_root / task_id.replace("/", "_")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "prompts.jsonl"
    log.write_text("", encoding="utf-8")
    print(f"\n[{task_id}] mode={mode}")

    source = "user"
    if not props:
        source = "llm"
        props = generate_properties(task, log)
        print("  no properties given, LLM wrote these:")
        for i, p in enumerate(props, 1):
            print(f"    P{i}: {p}")

    code = developer(task, log)
    (run_dir / "solution_v1.py").write_text(code, encoding="utf-8")
    (run_dir / "solution.py").write_text(code, encoding="utf-8")

    tests = qa_engineer(task, name, props, mode, log)
    save_tests(run_dir, tests, 1)

    ref_dir = None
    if reference:
        ref_dir = run_dir / "reference"
        ref_dir.mkdir(exist_ok=True)
        (ref_dir / "solution.py").write_text(reference, encoding="utf-8")

    attempts = []
    report_lines = [f"Task: {task_id}   mode: {mode}   model: {MODEL}",
                    f"Test validation: {'dataset reference solution' if reference else 'none (LLM pushback only)'}",
                    "", f"Properties (from {source}):"]
    report_lines += [f"  P{i}: {p}" for i, p in enumerate(props, 1)] + [""]
    regen_left = 1
    code_v, test_v, attempt, heal_rounds = 1, 1, 1, 0
    qa_fixes = {}       # property number -> times QA was asked to fix its test
    unresolved = set()  # properties whose test QA couldn't get right
    while True:
        res = judge(run_dir, name)
        if ref_dir:
            check_against_reference(res, run_dir, ref_dir, name)
        newly = set()
        for t in res["tests"]:
            pid = prop_number(t)
            if (t.get("kind") in ("bad_test", "test_error") and t["outcome"] != "passed"
                    and qa_fixes.get(pid, 0) >= MAX_TEST_FIX and pid not in unresolved):
                newly.add(pid)
        unresolved |= newly
        apply_unresolved(res, unresolved)
        if any(s["status"] == "NOT TESTED" for s in property_status(res["tests"], props)):
            res["passed"] = False
            res["bad_tests"] = True  # QA skipped a property, so regenerate tests
        attempts.append({"attempt": attempt, "code_version": code_v, "test_version": test_v,
                         **{k: v for k, v in res.items() if k != "output"}})
        (run_dir / f"result_{attempt}.txt").write_text(res["output"], encoding="utf-8")

        head, body = render_attempt(f"run {attempt} (code v{code_v}, tests v{test_v})", res, props)
        print("  " + head)
        for b in body:
            if "[PASS]" not in b:
                print("    " + b.replace("\n", "\n    "))
        report_lines += [head] + ["  " + b.replace("\n", "\n  ") for b in body] + [""]
        for pid in sorted(newly):
            msg = (f"P{pid} marked UNRESOLVED: its test still fails on the reference after "
                   f"{MAX_TEST_FIX} QA fixes, so it no longer counts towards the verdict")
            print(f"  ! {msg}")
            report_lines += [f"! {msg}", ""]

        if res["passed"] or mode == "basic":
            break
        attempt += 1
        if res["bad_tests"] and regen_left:
            regen_left -= 1
            note = "test file is broken or skips a property, QA regenerates the tests"
            tests = qa_engineer(task, name, props, mode, log)
            test_v += 1
            save_tests(run_dir, tests, test_v)
        elif bad := [t for t in res["tests"]
                     if t.get("kind") in ("bad_test", "test_error") and t["outcome"] != "unresolved"]:
            # fix the tests first, the code is only judged by tests we trust.
            # this has its own limit (MAX_TEST_FIX per property), so it doesn't eat developer rounds
            for pid in {prop_number(t) for t in bad}:
                qa_fixes[pid] = qa_fixes.get(pid, 0) + 1
            retry = max(qa_fixes[prop_number(t)] for t in bad)
            if any(t["kind"] == "bad_test" for t in bad):
                note = "some tests fail on the reference solution too, so the tests are wrong, sent to QA"
                reason = "these tests also fail on a known-correct reference implementation"
            else:
                note = "the failing tests broke on their own (not a code bug), sent to QA"
                reason = "the test raised an error by itself, the code never failed an assertion"
            feedback = feedback_from(bad, props)
            tests = qa_fix(task, name, props, mode, tests, reason, feedback, log, retry)
            test_v += 1
            save_tests(run_dir, tests, test_v)
        elif heal_rounds >= MAX_HEAL:
            break
        else:
            heal_rounds += 1
            feedback = feedback_from(res["tests"], props, res["output"])
            kind, out = developer_fix(task, code, feedback, log, can_blame_test=not reference,
                                      retry=heal_rounds)
            if kind == "code" and out.strip() == code.strip() and not reference:
                kind, out = "test_wrong", "developer returned the same code"
            if kind == "test_wrong":
                note = f"developer says the test is wrong ({out}), sent back to QA"
                tests = qa_fix(task, name, props, mode, tests, out, feedback, log, heal_rounds)
                test_v += 1
                save_tests(run_dir, tests, test_v)
            else:
                note = "developer rewrote the code"
                code = out
                code_v += 1
                (run_dir / f"solution_v{code_v}.py").write_text(code, encoding="utf-8")
                (run_dir / "solution.py").write_text(code, encoding="utf-8")
        print(f"  -> {note}")
        report_lines += [f"-> {note}", ""]

    final = attempts[-1]
    found = {"code_bug": [], "wrong_test": [], "broken_test": []}
    for a in attempts:
        for t in a["tests"]:
            if t["outcome"] == "passed":
                continue
            key = {"bad_test": "wrong_test", "test_error": "broken_test"}.get(t.get("kind"), "code_bug")
            m = re.match(r"test_p(\d+)", t["test"])
            pid = int(m.group(1)) if m else None
            exp_act = split_expected_actual(t.get("message", ""))
            found[key].append({
                "run": a["attempt"],
                "property": f"P{pid}: {props[pid - 1]}" if pid and pid <= len(props) else None,
                "test": t["test"],
                "input": test_input(t),
                "expected": exp_act[0] if exp_act else None,
                "actual": exp_act[1] if exp_act else None,
                "message": t.get("message", ""),
            })
    report = {
        "task_id": task_id,
        "mode": mode,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "verdict": "PASS" if final["passed"] else "FAIL",
        "attempts": len(attempts),
        "heal_rounds": heal_rounds,
        "code_versions": code_v,
        "test_versions": test_v,
        "property_source": source,
        "properties": [{k: v for k, v in s.items() if k != "tests"}
                       for s in property_status(final["tests"], props)],
        "unresolved_properties": [f"P{p}: {props[p - 1]}" for p in sorted(unresolved)],
        "code_bugs_found": found["code_bug"],
        "wrong_tests_found": found["wrong_test"],
        "broken_tests_found": found["broken_test"],
        "history": attempts,
    }
    labels = {"code_bug": "Failures caused by the code",
              "wrong_test": "Wrong tests (failed on the reference solution too)",
              "broken_test": "Broken tests (crashed by themselves)"}
    summary_start = len(report_lines)
    report_lines += ["Summary", f"  Verdict: {report['verdict']} after {len(attempts)} run(s)",
                     f"  Code rewritten: {code_v - 1} time(s), tests rewritten: {test_v - 1} time(s)"]
    if unresolved:
        report_lines.append(f"  Unresolved properties (not counted in the verdict): {len(unresolved)}")
        report_lines += [f"    {p}" for p in report["unresolved_properties"]]
    for key, label in labels.items():
        report_lines.append(f"  {label}: {len(found[key])}")
        for f in found[key]:
            report_lines.append(f"    run {f['run']}, {f['property'] or f['test']}")
            report_lines.append(f"      test:     {f['test']}")
            if f["input"] is not None:
                report_lines.append(f"      input:    {f['input']}")
            if f["expected"] is not None:
                report_lines += [f"      expected: {f['expected']}", f"      actual:   {f['actual']}"]
            else:
                report_lines.append(f"      error:    {f['message']}")

    # the judge agent reads everything above and writes the verdict report
    final_status = render_attempt("final run", final, props)
    facts = "\n".join([final_status[0], *final_status[1], "", *report_lines[summary_start:]])
    report["judge_report"] = judge_verdict(task, code, facts, report["verdict"], log)
    print("\n  Judge agent:\n    " + report["judge_report"].replace("\n", "\n    "))
    report_lines += ["", "Judge agent verdict", "  " + report["judge_report"].replace("\n", "\n  ")]

    (run_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_tasks(args):
    """Returns (task_id, prompt, function name, reference solution or None) for each task."""
    if args.prompt:
        if not args.name:
            sys.exit("--prompt needs --name")
        return [("custom", args.prompt + f"\nName the function `{args.name}`.", args.name, None)]

    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    idx = [args.task] if args.task is not None else range(args.n)
    # the reference is only used to check the generated tests, never shown to the agents
    return [(ds[i]["task_id"], ds[i]["prompt"], ds[i]["entry_point"],
             ds[i]["prompt"] + ds[i]["canonical_solution"]) for i in idx]


def ask_properties(task_id, task):
    """Let the user type properties for a task. Empty input means the LLM writes them."""
    if not sys.stdin.isatty():
        return []
    print(f"\n----- {task_id} -----\n{task.strip()}\n")
    print("Enter properties, one per line. Empty line to finish (or right away to let the LLM write them):")
    props = []
    while True:
        line = input("  > ").strip()
        if not line:
            return props
        props.append(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["basic", "advanced"], default="basic")
    ap.add_argument("--task", type=int, help="HumanEval index")
    ap.add_argument("--n", type=int, default=5, help="run the first n HumanEval tasks")
    ap.add_argument("--prompt", help="custom natural language prompt")
    ap.add_argument("--name", help="function name for a custom prompt")
    ap.add_argument("--property", action="append", help="property to test, repeat for more")
    args = ap.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY missing, add it to .env")
    if args.property and not (args.prompt or args.task is not None):
        sys.exit("--property only works with a single task (--task or --prompt)")

    out_root = RUNS_DIR / args.mode
    reports = []
    for task_id, task, name, reference in load_tasks(args):
        props = args.property or ask_properties(task_id, task)
        reports.append(run_task(task_id, task, name, props, args.mode, out_root, reference))

    passed = sum(r["verdict"] == "PASS" for r in reports)
    print(f"\n{passed}/{len(reports)} passed ({args.mode} mode)")
    (out_root / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
