# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Full-scan the repo with Radon, import-linter and SonarQube on BASE and on
the working tree, list the issues the working tree adds, score the diff and
time each tool.

usage: python ci/code_health_check.py BASE_REF

Scope, Sonar issues excluded from scoring and the scoring (weights, zero lines, curve,
block thresholds) live in ci/code_health_check.toml, read from BASE;
this script and the workflow themselves run from the PR.

- Radon: functions with CC > cc_max that are new or got more complex.
- import-linter: new sibling-package cycles (acyclic_siblings on paddlefleet),
  plus new module-level `from m import name` edges on a cycle of
  module-level imports (those can raise ImportError; plain module imports
  on a cycle cannot).
- SonarQube: BASE then HEAD are analysed into one throwaway project; issues
  in the new code period (default "previous version") are the new ones.
  Needs SONAR_HOST_URL / SONAR_TOKEN (fails without them); SONAR_SCANNER
  overrides the sonar-scanner found on PATH.

The markdown report goes to stdout and to $GITHUB_STEP_SUMMARY when set.
Exits 1 on verdict "block".
"""

import ast
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from fnmatch import fnmatch
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
CONFIG = "ci/code_health_check.toml"
PKG_DIR, PKG = "src", "paddlefleet"
SONAR_HOST = os.environ.get("SONAR_HOST_URL", "").rstrip("/")
SONAR_TOKEN = os.environ.get("SONAR_TOKEN", "")
SONAR_SCANNER = os.environ.get("SONAR_SCANNER") or shutil.which("sonar-scanner")
LIMIT = 200  # max issues listed per tool


def in_scope(cfg, rel):
    sc = cfg["scope"]
    return any(
        rel.startswith(d.rstrip("/") + "/") for d in sc["sources"]
    ) and not any(
        fnmatch(rel, p) or fnmatch(rel, p.removeprefix("**/"))
        for p in sc["exclude"]
    )


def py_files(cfg, root):
    for d in cfg["scope"]["sources"]:
        for f in sorted((root / d).rglob("*.py")):
            rel = f.relative_to(root).as_posix()
            if in_scope(cfg, rel):
                yield rel, f


def added_lines(cfg, base_sha):
    """{path: {line, ...}} of in-scope .py lines added since base_sha."""
    out = subprocess.check_output(
        ["git", "diff", "-U0", "--no-color", "--no-renames", base_sha, "HEAD"],
        cwd=ROOT,
        text=True,
    )
    added, path = {}, None
    for line in out.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
            if path and not (path.endswith(".py") and in_scope(cfg, path)):
                path = None
        elif line.startswith("@@") and path:
            start, count = re.match(
                r"@@ -\S+ \+(\d+)(?:,(\d+))?", line
            ).groups()
            start, count = int(start), int(1 if count is None else count)
            added.setdefault(path, set()).update(range(start, start + count))
    return added


# ---------------------------------------------------------------- radon
def radon(cfg, root):
    """{"file::func": [cc, first_line, last_line]}"""
    from radon.complexity import cc_visit
    from radon.visitors import Function

    cc = {}
    for rel, f in py_files(cfg, root):
        try:
            blocks = cc_visit(f.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError):
            continue
        for b in blocks:
            key = f"{rel}::{b.fullname}"
            if isinstance(b, Function) and b.complexity > cc.get(key, [0])[0]:
                cc[key] = [b.complexity, b.lineno, b.endline]
    return cc


def radon_new(cfg, base, head):
    """Keys of functions this tree made complex."""
    cc_max = cfg["radon"]["cc_max"]
    return [
        k
        for k, (n, _, _) in sorted(head.items())
        if n > cc_max and n > base.get(k, [0])[0]
    ]


# --------------------------------------------------------- import-linter
INI = f"""[importlinter]
root_packages =
    {PKG}
exclude_type_checking_imports = True

[importlinter:contract:acyclic]
name = {PKG} sibling packages are acyclic
type = acyclic_siblings
ancestors =
    {PKG}
"""


def imports(root):
    # grimp caches module lookups per process, so each tree gets its own.
    out = subprocess.run(
        [sys.executable, __file__, "--imports", str(root)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)


def _toplevel_import_lines(path):
    """Lines of imports that run at import time (not inside a function)."""
    lines = set()

    def walk(node, in_func):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)) and not in_func:
                lines.add(child.lineno)
            walk(
                child,
                in_func
                or isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
                ),
            )

    try:
        walk(
            ast.parse(path.read_text(encoding="utf-8", errors="replace")), False
        )
    except SyntaxError:
        pass
    return lines


def _imports_name(line, target):
    """True when `line` binds a name from module `target`, not the module:
    `from target import name` fails while target is half-initialised,
    `import target` / `from pkg import target` do not."""
    if line.startswith("import "):
        return False
    names = line.split(" import ", 1)[-1].strip("()\\ ").split(",")
    # ponytail: only the first line of a parenthesised import is seen, so a
    # multi-line `from pkg import (submodule,` counts as a name import.
    return target.rsplit(".", 1)[-1] not in {
        n.split()[0] for n in names if n.split()
    }


def _module_cycle_edges(graph, tmp):
    """Edges `a -> b` where a does a module-level `from b import name` and
    a, b lie on one cycle of module-level imports: some import order then
    raises ImportError (name not yet defined in the half-initialised b)."""
    edges, named = {}, set()
    for m in graph.modules:
        f = tmp / (m.replace(".", "/") + ".py")
        f = f if f.exists() else tmp / m.replace(".", "/") / "__init__.py"
        top = _toplevel_import_lines(f)
        edges[m] = []
        for t in graph.find_modules_directly_imported_by(m):
            ds = [
                d
                for d in graph.get_import_details(importer=m, imported=t)
                if d["line_number"] in top
            ]
            if ds:
                edges[m].append(t)
            if any(_imports_name(d["line_contents"], t) for d in ds):
                named.add((m, t))
    return _on_cycle(edges, named)


def _on_cycle(edges, named):
    """Edges `a -> b` of `named` where b imports its way back to a."""

    def reaches(src, dst):
        seen, todo = {src}, [src]
        while todo:
            for w in edges.get(todo.pop(), ()):
                if w == dst:
                    return True
                if w not in seen:
                    seen.add(w)
                    todo.append(w)
        return False

    return sorted(f"{a} -> {b}" for a, b in named if reaches(b, a))


def _imports_child(root):
    """Print {"pkg": {"pkg|childA -> childB": "importer -> imported"},
    "mod": [name-import edges on module-level cycles]}."""
    import grimp
    from importlinter import configuration
    from importlinter.application import use_cases

    configuration.configure()
    tmp = Path(tempfile.mkdtemp(prefix="importlinter-"))
    try:
        # grimp skips directories without __init__.py; add them in a copy.
        shutil.copytree(
            Path(root) / PKG_DIR / PKG,
            tmp / PKG,
            ignore=shutil.ignore_patterns("__pycache__", "*.so"),
        )
        for d in {f.parent for f in (tmp / PKG).rglob("*.py")}:
            for p in [d, *d.parents]:
                if p == tmp:
                    break
                (p / "__init__.py").touch()
        (tmp / ".importlinter").write_text(INI)
        os.chdir(tmp)
        sys.path.insert(0, str(tmp))
        opts = use_cases.read_user_options(".importlinter")
        use_cases._register_contract_types(opts)
        report = use_cases.create_report(opts, cache_dir=None)
        _, check = next(report.get_contracts_and_checks())
        cycles = {}
        for pkg, pairs in check.metadata["cycle_breakers_by_package"].items():
            depth = pkg.count(".") + 2
            for a, b in sorted(pairs):
                sa, sb = (".".join(m.split(".")[:depth]) for m in (a, b))
                cycles.setdefault(f"{pkg}|{sa} -> {sb}", f"{a} -> {b}")
        graph = grimp.build_graph(
            PKG, exclude_type_checking_imports=True, cache_dir=None
        )
        print(
            json.dumps({"pkg": cycles, "mod": _module_cycle_edges(graph, tmp)})
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def imports_new(base, head):
    """(new package cycles {key: example}, new name-import cycle edges)"""
    # ponytail: breakers come from a heuristic feedback-arc-set, so an
    # unrelated graph change can occasionally move one; keyed at sibling
    # level to keep that rare.
    pkg = {k: v for k, v in sorted(head["pkg"].items()) if k not in base["pkg"]}
    return pkg, sorted(set(head["mod"]) - set(base["mod"]))


# ------------------------------------------------------------- sonarqube
def api(path, method="GET", **params):
    q = urllib.parse.urlencode(params)
    url, data = (
        (f"{SONAR_HOST}{path}?{q}", None)
        if method == "GET"
        else (f"{SONAR_HOST}{path}", q.encode())
    )
    auth = base64.b64encode(f"{SONAR_TOKEN}:".encode()).decode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Authorization": f"Basic {auth}"},
    )
    # The server is on the intranet; never send the token through a proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=60) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def sonar_drop(key):
    try:
        api("/api/projects/delete", "POST", project=key)
    except OSError:
        pass  # did not exist / no admin right: the next analysis still works


def sonar(cfg, root, key, version):
    """Analyse `root` into project `key` and wait for the server."""
    work = tempfile.mkdtemp(prefix="sonar-")
    env = {
        k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")
    }
    env.setdefault("SONAR_SCANNER_OPTS", "-Xmx4g")
    cmd = [
        SONAR_SCANNER,
        f"-Dsonar.host.url={SONAR_HOST}",
        f"-Dsonar.projectKey={key}",
        f"-Dsonar.projectVersion={version}",
        f"-Dsonar.projectBaseDir={root}",
        f"-Dsonar.working.directory={work}",
        "-Dsonar.sources="
        + ",".join(d for d in cfg["scope"]["sources"] if (root / d).is_dir()),
        "-Dsonar.exclusions=" + ",".join(cfg["scope"]["exclude"]),
        "-Dsonar.scm.disabled=true",
        "-Dsonar.python.version=3.10,3.11,3.12,3.13",
        "-Dsonar.sourceEncoding=UTF-8",
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode:
            sys.stderr.write(proc.stdout[-3000:] + proc.stderr[-3000:])
            raise RuntimeError("sonar-scanner failed")
        task = re.search(
            r"^ceTaskId=(\S+)",
            (Path(work) / "report-task.txt").read_text(),
            re.MULTILINE,
        ).group(1)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    deadline = time.time() + 1800
    while time.time() < deadline:
        status = api("/api/ce/task", id=task)["task"]["status"]
        if status not in ("PENDING", "IN_PROGRESS"):
            if status != "SUCCESS":
                raise RuntimeError(f"sonar server analysis {status}")
            return
        time.sleep(3)
    raise RuntimeError("sonar server analysis timed out")


def flagged_code(i):
    """Source text Sonar flagged for issue i, read from the HEAD tree."""
    r = i.get("textRange")
    if not r:
        return ""
    try:
        text = (ROOT / i["path"]).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()[r["startLine"] - 1 : r["endLine"]]
    if not lines:
        return ""
    lines[-1] = lines[-1][: r["endOffset"]]
    lines[0] = lines[0][r["startOffset"] :]
    return "\n".join(lines)


def is_excluded(sc, i):
    """True when issue i is left out of scoring by [sonar] in the config."""
    name = re.search(r'"([^"]+)"', i["message"])
    return any(
        i["rule"] == x["rule"]
        and any(fnmatch(i["path"], p) for p in x.get("paths", ["*"]))
        and ("code" not in x or re.search(x["code"], i.get("code", "")))
        for x in sc["exclude"]
    ) or bool(
        i["rule"] in sc["naming_rules"]
        and name
        and re.match(sc["naming_extra"], name[1])
    )


def sonar_new(cfg, key):
    """(kept, excluded) new issues. Every rule stays active on the server;
    excluded issues are reported apart and not scored."""
    kept, excluded, page = [], [], 1
    while True:
        r = api(
            "/api/issues/search",
            componentKeys=key,
            resolved="false",
            inNewCodePeriod="true",
            ps=500,
            p=page,
        )
        for i in r["issues"]:
            i["path"] = i["component"].split(":", 1)[1]
            i["code"] = flagged_code(i)
            (excluded if is_excluded(cfg["sonar"], i) else kept).append(i)
        if page * 500 >= r["paging"]["total"]:
            for x in (kept, excluded):
                x.sort(key=lambda i: (i["path"], i.get("line", 0)))
            return kept, excluded
        page += 1


def sonar_line(i):
    return (
        f"`{i['path']}:{i.get('line', '-')}` [{i['type']} "
        f"{i.get('severity', '')} {i['rule']}] {i['message']}"
    )


def sonar_duplicated(key, added):
    """{path: [line, ...]} of added lines that sit in a Sonar duplicated block."""
    dup = {}
    for path, lines in sorted(added.items()):
        try:
            src = api(
                "/api/sources/lines",
                key=f"{key}:{path}",
                to=max(lines),
            )["sources"]
        except OSError:
            continue  # file not analysed (e.g. empty)
        hit = [
            x["line"] for x in src if x.get("duplicated") and x["line"] in lines
        ]
        if hit:
            dup[path] = hit
    return dup


def spans(ns):
    """[3, 4, 5, 9] -> "3-5, 9"."""
    out = []
    for n in sorted(ns):
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


# -------------------------------------------------------------- scoring
def item_score(cfg, value, zero):
    k = cfg["score"]["curve_k"]
    x = min(value / zero, 1.0)
    return 100 * (1 - math.log1p(k * x) / math.log1p(k))


def score(cfg, m):
    """m: metric values -> (rows, total, verdict).
    rows: [(name, value text, item score, verdict, reason)]"""
    sc = cfg["score"]
    rows, total = [], 0.0
    spec = (
        ("bug", "Bug + 漏洞 / 千行", lambda v: f"{v:.2f}"),
        ("smell", "可维护性问题 / 千行", lambda v: f"{v:.1f}"),
        ("dup", "代码重复率", lambda v: f"{v:.1%}"),
        ("cc", "分支复杂度", lambda v: f"{v:.1%}"),
        ("cycle", "循环依赖", lambda v: f"{v:.1%}"),
    )
    for key, name, fmt in spec:
        v, c = m[key], sc[key]
        s = item_score(cfg, v, c["zero"])
        total += c["weight"] * s
        reason = m.get(key + "_why", "")
        if key == "cycle":
            lv = "block" if m["mod_cycles"] else "pass"
        else:
            lv = "pass" if v <= c["block"] else "block"
            if key == "bug" and m.get("bug_severe"):
                lv = "block"
                reason = f"新增 {m['bug_severe']} 条 {'/'.join(c['block_severities'])}"
        rows.append((name, fmt(v), s, lv, reason))
    blocked = total < sc["total"]["block"] or any(r[3] == "block" for r in rows)
    return rows, total, "block" if blocked else "pass"


def selftest():
    """Numbers from the scoring doc (section 9.2)."""
    with open(ROOT / CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    assert round(item_score(cfg, 0.01, 1), 1) == 92.4
    assert round(item_score(cfg, 0.1, 1), 1) == 59.6
    # #2268: only 19.0 smells / 1k lines -> 96.1, mergeable.
    m = {"bug": 0, "smell": 19.0, "dup": 0, "cc": 0, "cycle": 0}
    m |= {"pkg_cycles": 0, "mod_cycles": 0}
    rows, total, verdict = score(cfg, m)
    assert round(total, 1) == 96.1 and verdict == "pass", (total, verdict)
    # #2251: 11.9 smells, dup 2.4%, cc 19.6%, 3/4 packages in new cycles.
    m.update(smell=11.9, dup=0.024, cc=0.196, cycle=0.75, pkg_cycles=2)
    rows, total, verdict = score(cfg, m)
    assert round(total, 1) == 75.3 and verdict == "block", (total, verdict)
    assert spans([9, 3, 4, 5]) == "3-5, 9"
    # Exclusions: by rule, by rule + path, and the naming relaxation.
    ex = lambda rule, path, msg="m", code="": is_excluded(
        cfg["sonar"], {"rule": rule, "path": path, "message": msg, "code": code}
    )
    # S1244 only against a bare whole-number literal.
    assert ex("python:S1244", "src/a.py", code="self._wd != 0.0")
    assert ex("python:S1244", "src/a.py", code="mask = input == 0.0")
    assert ex("python:S1244", "src/a.py", code="w = s if alpha == 2.0 else t")
    assert not ex("python:S1244", "src/a.py", code="a == 0.1")
    assert not ex("python:S1244", "src/a.py", code="s != 1.0 / (\n  d**0.5\n)")
    assert not ex("python:S1244", "src/a.py", code="x == 1.0e-5")
    assert ex("python:S2245", "src/paddlefleet/examples/x.py")
    assert not ex("python:S2245", "src/paddlefleet/data/x.py")
    assert ex("python:S5778", "src/paddlefleet/ops/test_x.py")
    assert not ex("python:S1226", "src/paddlefleet/triton_ops/x.py")
    assert ex("python:S117", "src/a.py", 'Rename "BLOCK_M" to match')
    assert not ex("python:S117", "src/a.py", 'Rename "myValue" to match')
    # Only `from m import name` can fail on a half-initialised m.
    assert _imports_name("from a.b import f", "a.b")
    assert _imports_name("from .b import f, g as h", "a.b")
    assert not _imports_name("from a import b", "a.b")
    assert not _imports_name("from . import c, b as x", "a.b")
    assert not _imports_name("import a.b as x", "a.b")
    print("selftest ok")


# ------------------------------------------------------------------ main
VERDICT = {"pass": "可合入", "block": "不予合入"}


def timed(fn, *args):
    t = time.monotonic()
    res = fn(*args)
    return res, time.monotonic() - t


def load_config(base_sha):
    """The base branch's config, so a PR cannot loosen it in the toml."""
    try:
        text = subprocess.check_output(
            ["git", "show", f"{base_sha}:{CONFIG}"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:  # the PR that adds the config
        text = (ROOT / CONFIG).read_text()
    return tomllib.loads(text)


def git_sha(ref):
    return subprocess.check_output(
        ["git", "rev-parse", ref], cwd=ROOT, text=True
    ).strip()


def main(base_ref):
    if not (SONAR_HOST and SONAR_TOKEN):
        sys.exit("SONAR_HOST_URL / SONAR_TOKEN not set")
    # In CI HEAD is a local merge commit; name the PR's own head instead.
    base_sha = git_sha(base_ref)
    head_sha = os.environ.get("PR_HEAD_SHA") or git_sha("HEAD")
    cfg = load_config(base_sha)
    added = {}
    for path, lines in added_lines(cfg, base_sha).items():
        src = (ROOT / path).read_text(encoding="utf-8", errors="replace")
        src = src.splitlines()
        code = {n for n in lines if n <= len(src) and src[n - 1].strip()}
        if code:  # count code lines, not blank ones
            added[path] = code
    n_lines = sum(map(len, added.values()))
    denom = max(n_lines, cfg["score"]["min_lines"]) / 1000
    m, times, found, excluded = {}, [], {}, []

    tmp = Path(tempfile.mkdtemp(prefix="code-health-"))
    base = tmp / "base"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(base), base_sha],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    try:
        # Radon: share of changed functions this diff made complex.
        (cb, tb), (ch, th) = timed(radon, cfg, base), timed(radon, cfg, ROOT)
        times.append(("Radon", tb, th))
        changed = [
            k
            for k, (_, lo, hi) in ch.items()
            if any(lo <= n <= hi for n in added.get(k.split("::")[0], ()))
        ]
        bad = [k for k in radon_new(cfg, cb, ch) if k in changed]
        m["cc"] = len(bad) / len(changed) if changed else 0.0
        m["cc_why"] = f"{len(bad)}/{len(changed)} 个函数"
        found["分支复杂度"] = [
            f"`{k}` CC {ch[k][0]} > {cfg['radon']['cc_max']}"
            + (f"（原 {cb[k][0]}）" if k in cb else "（新函数）")
            for k in bad
        ]

        # import-linter: share of changed packages caught in a new cycle.
        (ib, tb), (ih, th) = timed(imports, base), timed(imports, ROOT)
        times.append(("import-linter", tb, th))
        pkg, mod = imports_new(ib, ih)
        ends = {x for k in pkg for x in k.split("|")[1].split(" -> ")}
        ends |= {x for e in mod for x in e.split(" -> ")}
        pkgs = {
            ".".join(Path(p).with_suffix("").parts[1:-1])
            for p in added
            if p.startswith(f"{PKG_DIR}/{PKG}/")
        }
        hit = [
            p
            for p in pkgs
            if any(
                p == e or p.startswith(e + ".") or e.startswith(p + ".")
                for e in ends
            )
        ]
        m["cycle"] = len(hit) / len(pkgs) if pkgs else 0.0
        m["cycle_why"] = f"{len(hit)}/{len(pkgs)} 个包"
        m["mod_cycles"] = len(mod)
        found["循环依赖"] = [
            f"包级环 `{k.split('|')[1]}`（如 `{v}`）" for k, v in pkg.items()
        ] + [
            f"模块级环中按名字导入（某些 import 顺序下 ImportError）`{e}`"
            for e in mod
        ]

        key = "PaddleFleet-ci-" + os.environ.get("PR_ID", head_sha[:12])
        sonar_drop(key)
        try:
            _, tb = timed(sonar, cfg, base, key, f"base-{base_sha[:12]}")
            _, th = timed(sonar, cfg, ROOT, key, f"head-{head_sha[:12]}")
            times.append(("SonarQube", tb, th))
            kept, excluded = sonar_new(cfg, key)
            dup = sonar_duplicated(key, added)
        finally:
            sonar_drop(key)
        bugs = [i for i in kept if i["type"] in ("BUG", "VULNERABILITY")]
        smells = [i for i in kept if i not in bugs]
        sev = cfg["score"]["bug"]["block_severities"]
        m["bug"] = len(bugs) / denom
        m["bug_why"] = f"{len(bugs)} 条"
        m["bug_severe"] = sum(i.get("severity") in sev for i in bugs)
        m["smell"] = len(smells) / denom
        m["smell_why"] = f"{len(smells)} 条"
        n_dup = sum(map(len, dup.values()))
        m["dup"] = n_dup / n_lines if n_lines else 0.0
        m["dup_why"] = f"{n_dup}/{n_lines} 行"
        found["Bug + 漏洞"] = [sonar_line(i) for i in bugs]
        found["可维护性"] = [sonar_line(i) for i in smells]
        found["代码重复率"] = [
            f"`{p}` 第 {spans(ns)} 行（{len(ns)} 行）" for p, ns in dup.items()
        ]
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(base)],
            cwd=ROOT,
            capture_output=True,
        )
        shutil.rmtree(tmp, ignore_errors=True)

    rows, total, verdict = score(cfg, m)
    out = [
        f"## Code Health Check：`{head_sha[:12]}` vs base `{base_sha[:12]}`",
        "",
        f"**判定：{VERDICT[verdict]}**，综合得分 {total:.1f}；"
        f"新增代码 {n_lines} 行"
        f"（千行指标分母至少按 {cfg['score']['min_lines']} 行）。"
        f"规则见 `{CONFIG}`。",
        "",
        "| 维度 | diff 值 | 单项分 | 判定 | 说明 |",
        "|---|---|---|---|---|",
        *(
            f"| {n} | {v} | {s:.1f} | {VERDICT[lv]} | {why} |"
            for n, v, s, lv, why in rows
        ),
        "",
        "| 工具 | base 全量扫描 | head 全量扫描 | 合计 |",
        "|---|---|---|---|",
        *(f"| {n} | {b:.1f}s | {h:.1f}s | {b + h:.1f}s |" for n, b, h in times),
        "",
    ]
    # Issues grouped by the 5 scored dimensions, in table order.
    for d in ("Bug + 漏洞", "可维护性", "代码重复率", "分支复杂度", "循环依赖"):
        new = found[d]
        count = f"{n_dup} 行" if d == "代码重复率" else f"{len(new)} 条"
        out += [f"### {d}：新增 {count}", ""]
        out += [f"- {x}" for x in new[:LIMIT]]
        if len(new) > LIMIT:
            out.append(f"- ... 另有 {len(new) - LIMIT} 条")
        out.append("")
    if excluded:
        out += [
            f"<details><summary>已剔除规则：{len(excluded)} 条</summary>",
            "",
            *(f"- {sonar_line(i)}" for i in excluded[:LIMIT]),
            "",
            "</details>",
        ]
    report = "\n".join(out)
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")
    return 1 if verdict == "block" else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--imports"]:
        _imports_child(sys.argv[2])
    elif sys.argv[1:] == ["--selftest"]:
        selftest()
    elif len(sys.argv) == 2:
        sys.exit(main(sys.argv[1]))
    else:
        sys.exit(__doc__)
