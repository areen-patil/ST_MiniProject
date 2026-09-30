"""Developer -> QA engineer -> Judge pipeline with a self-healing loop.

Usage:
    python pipeline.py --mode basic --task 0
    python pipeline.py --mode advanced --n 10
    python pipeline.py --mode advanced --prompt "Write a function is_prime(n) ..."
"""
import argparse
import json
import os
import re
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
MAX_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
MAX_HEAL = 3
TEST_TIMEOUT = 90
RUNS_DIR = Path("runs")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DEV_SYSTEM = (
    "You are a Python developer. Reply with one complete Python module inside a "
    "```python block. Include needed imports. No explanations, no example usage, "
    "no input() or print() calls."
)
QA_BASIC_SYSTEM = (
    "You are a QA engineer. Write a pytest file for the given function. Start with "
    "`from solution import {name}`. Use plain assert statements, several cases, "
    "covering every branch and statement, including edge cases and error paths. "
    "Reply with one ```python block only."
)
QA_ADV_SYSTEM = (
    "You are a QA engineer. Write a pytest file using the hypothesis library for the "
    "given function. Start with `from solution import {name}`. Use @given with "
    "suitable strategies and @settings(max_examples=100, deadline=None). Test "
    "properties and invariants that must hold for any valid input (output type, "
    "idempotence, agreement with a simple reference implementation written in the "
    "test, boundary values). Only generate inputs the task description allows. "
    "Reply with one ```python block only."
)


def call_gemini(system, user, log, tag):
    """Single Gemini call; logs prompt + settings to the run's prompts.jsonl."""
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_output_tokens=MAX_TOKENS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    text = None
    for attempt in range(5):
        try:
            text = client.models.generate_content(model=MODEL, contents=user, config=cfg).text
            break
        except Exception as e:
            wait = 5 * 2 ** attempt
            print(f"  gemini error ({e.__class__.__name__}), retrying in {wait}s")
            time.sleep(wait)
    if text is None:
        raise RuntimeError("Gemini call failed after retries")

    entry = {
        "agent": tag,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_output_tokens": MAX_TOKENS,
        "system_prompt": system,
        "user_prompt": user,
        "response": text,
    }
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return text


def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip() + "\n"


def developer(task, log, prev_code=None, feedback=None):
    if prev_code is None:
        user = f"Task:\n{task}"
    else:
        user = (
            f"Task:\n{task}\n\nYour previous code:\n```python\n{prev_code}```\n\n"
            f"It failed these tests:\n{feedback}\n\n"
            "Fix the logic and return the full corrected module."
        )
    return extract_code(call_gemini(DEV_SYSTEM, user, log, "developer"))


def qa_engineer(task, code, name, mode, log):
    system = (QA_BASIC_SYSTEM if mode == "basic" else QA_ADV_SYSTEM).format(name=name)
    user = f"Task:\n{task}\n\nCode under test:\n```python\n{code}```"
    return extract_code(call_gemini(system, user, log, f"qa_{mode}"))


def falsifying_example(output):
    m = re.search(r"Falsifying example:.*?(?:\n\s*\n|\Z)", output, re.DOTALL)
    return m.group(0).strip() if m else None


def judge(run_dir, mode):
    """Run pytest on solution.py + test_solution.py in run_dir."""
    cmd = [sys.executable, "-m", "pytest", "test_solution.py", "-x", "-q", "--tb=short", "-p", "no:cacheprovider"]
    if mode == "basic":
        cmd += ["--cov=solution", "--cov-report=term-missing"]
    try:
        p = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True, timeout=TEST_TIMEOUT)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        out, rc = f"Timed out after {TEST_TIMEOUT}s (possible infinite loop)", -1

    cov = None
    m = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", out)
    if m:
        cov = int(m.group(1))

    result = {"passed": rc == 0, "returncode": rc, "coverage": cov, "output": out}
    result["bad_tests"] = rc in (2, 4)  # collection/usage error, tests themselves are broken
    fe = falsifying_example(out)
    if fe:
        result["falsifying_example"] = fe
    return result


def feedback_from(result):
    tail = "\n".join(result["output"].splitlines()[-40:])
    fe = result.get("falsifying_example")
    return f"{fe}\n\n{tail}" if fe else tail


def run_task(task_id, task, name, mode, out_root):
    run_dir = out_root / task_id.replace("/", "_")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "prompts.jsonl"
    log.write_text("", encoding="utf-8")
    print(f"\n[{task_id}] mode={mode}")

    code = developer(task, log)
    (run_dir / "solution_v1.py").write_text(code, encoding="utf-8")
    (run_dir / "solution.py").write_text(code, encoding="utf-8")

    tests = qa_engineer(task, code, name, mode, log)
    (run_dir / "test_solution.py").write_text(tests, encoding="utf-8")

    attempts = []
    regen_left = 1
    version = 1
    while True:
        res = judge(run_dir, mode)
        attempts.append({"version": version, **{k: v for k, v in res.items() if k != "output"}})
        (run_dir / f"result_v{version}.txt").write_text(res["output"], encoding="utf-8")
        status = "PASS" if res["passed"] else "FAIL"
        print(f"  v{version}: {status}" + (f" cov={res['coverage']}%" if res["coverage"] is not None else ""))

        if res["passed"] or mode == "basic":
            break
        if res["bad_tests"] and regen_left:
            regen_left -= 1
            print("  test file is broken, regenerating tests")
            tests = qa_engineer(task, code, name, mode, log)
            (run_dir / "test_solution.py").write_text(tests, encoding="utf-8")
            continue
        if version > MAX_HEAL:
            break

        version += 1
        code = developer(task, log, prev_code=code, feedback=feedback_from(res))
        (run_dir / f"solution_v{version}.py").write_text(code, encoding="utf-8")
        (run_dir / "solution.py").write_text(code, encoding="utf-8")

    final = attempts[-1]
    report = {
        "task_id": task_id,
        "mode": mode,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "verdict": "PASS" if final["passed"] else "FAIL",
        "attempts": len(attempts),
        "heal_rounds": version - 1,
        "coverage": final["coverage"],
        "history": attempts,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_tasks(args):
    if args.prompt:
        if not args.name:
            sys.exit("--name (function name) is required with --prompt")
        return [("custom", args.prompt + f"\nName the function `{args.name}`.", args.name)]

    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    if args.task is not None:
        rows = [ds[args.task]]
    else:
        rows = [ds[i] for i in range(args.n)]
    return [(r["task_id"], r["prompt"], r["entry_point"]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["basic", "advanced"], default="basic")
    ap.add_argument("--task", type=int, help="HumanEval index")
    ap.add_argument("--n", type=int, default=5, help="run the first n HumanEval tasks")
    ap.add_argument("--prompt", help="custom natural language prompt")
    ap.add_argument("--name", help="function name for a custom prompt")
    args = ap.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY missing, add it to .env")

    out_root = RUNS_DIR / args.mode
    reports = [run_task(tid, t, n, args.mode, out_root) for tid, t, n in load_tasks(args)]

    passed = sum(r["verdict"] == "PASS" for r in reports)
    print(f"\n{passed}/{len(reports)} passed ({args.mode} mode)")
    covs = [r["coverage"] for r in reports if r["coverage"] is not None]
    if covs:
        print(f"mean coverage: {sum(covs) / len(covs):.1f}%")
    (out_root / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
