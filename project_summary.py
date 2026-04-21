"""
프로젝트 & 파이썬 환경 통합 정리 도구
- 설치된 Python 환경 / 패키지 현황
- 프로젝트 .py 파일별 임포트 · 함수 · 클래스 요약
"""

import os, ast, sys, subprocess
from pathlib import Path
from collections import defaultdict


# ── 터미널 색상 ───────────────────────────────────────────────────────────────
class C:
    R = "\033[0m"; BOLD = "\033[1m"
    CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    RED = "\033[91m"; GRAY = "\033[90m"; BLUE = "\033[94m"
    MAGENTA = "\033[95m"; WHITE = "\033[97m"

def b(t):  return f"{C.BOLD}{t}{C.R}"
def cy(t): return f"{C.CYAN}{t}{C.R}"
def g(t):  return f"{C.GREEN}{t}{C.R}"
def y(t):  return f"{C.YELLOW}{t}{C.R}"
def gr(t): return f"{C.GRAY}{t}{C.R}"
def m(t):  return f"{C.MAGENTA}{t}{C.R}"
def bl(t): return f"{C.BLUE}{t}{C.R}"
def rd(t): return f"{C.RED}{t}{C.R}"

# ── 라이브러리 카테고리 정의 ──────────────────────────────────────────────────
CATEGORIES = {
    "🤖 AI / ML": [
        "tensorflow", "tf_keras", "keras", "torch", "torchvision",
        "transformers", "sentence_transformers", "sklearn", "scikit_learn",
        "ultralytics", "mediapipe", "spacy", "huggingface_hub", "safetensors",
        "tokenizers", "einops",
    ],
    "👁 컴퓨터 비전": [
        "cv2", "PIL", "pillow", "imageio", "skimage", "opencv",
    ],
    "📊 데이터 처리": [
        "numpy", "pandas", "polars", "pyarrow", "scipy", "statsmodels",
        "patsy", "joblib", "threadpoolctl", "ml_dtypes",
    ],
    "📈 시각화": [
        "matplotlib", "seaborn", "plotly", "pydeck", "altair",
    ],
    "🌐 웹 / 네트워크": [
        "requests", "urllib", "httpx", "httpcore", "selenium", "websocket",
        "aiohttp", "flask", "fastapi", "werkzeug", "starlette",
    ],
    "📓 Jupyter / 노트북": [
        "jupyter", "jupyterlab", "ipython", "ipykernel", "ipywidgets",
        "nbformat", "nbconvert", "notebook",
    ],
    "🚀 앱 / 서비스": [
        "streamlit", "gradio", "dash",
    ],
    "💾 파일 / 시스템": [
        "os", "sys", "pathlib", "shutil", "glob", "io", "csv", "json",
        "yaml", "toml", "zipfile", "tarfile",
    ],
    "⏱ 시간 / 스케줄": [
        "time", "datetime", "timeit", "calendar", "arrow",
    ],
    "⚙ 멀티스레드 / 비동기": [
        "threading", "multiprocessing", "concurrent", "asyncio", "trio",
        "queue", "subprocess",
    ],
    "🔢 수학 / 통계": [
        "math", "random", "statistics", "sympy", "mpmath", "decimal",
    ],
    "🧰 유틸 / 기타": [],   # 위 분류 외 나머지
}

def categorize(pkg_name):
    low = pkg_name.lower().replace("-", "_")
    for cat, members in CATEGORIES.items():
        if low in members or any(low.startswith(m) for m in members):
            return cat
    return "🧰 유틸 / 기타"


# ═══════════════════════════════════════════════════════════════════
# 1. Python 환경 정보
# ═══════════════════════════════════════════════════════════════════

def show_env():
    print(f"\n{cy('═'*70)}")
    print(f" {b('🐍 Python 환경')}")
    print(cy('═'*70))

    exe = Path(sys.executable)
    ver = sys.version.split()[0]
    print(f"  버전    : {b(ver)}")
    print(f"  경로    : {gr(str(exe))}")
    print(f"  site-packages: {gr(str(Path(exe).parent.parent / 'Lib' / 'site-packages'))}")

    # pip list
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=columns"],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().splitlines()[2:]  # 헤더 제거
        packages = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                packages[parts[0].lower()] = (parts[0], parts[1])
    except Exception as e:
        print(rd(f"  pip list 오류: {e}"))
        return

    # 카테고리별 분류
    by_cat = defaultdict(list)
    for key, (name, ver_) in packages.items():
        cat = categorize(key)
        by_cat[cat].append((name, ver_))

    print(f"\n  총 {b(len(packages))}개 패키지 설치됨\n")

    cat_order = list(CATEGORIES.keys())
    for cat in cat_order:
        items = by_cat.get(cat, [])
        if not items:
            continue
        items.sort(key=lambda x: x[0].lower())
        print(f"  {b(cat)}  ({len(items)}개)")
        row = ""
        for i, (name, ver_) in enumerate(items):
            entry = f"{g(name)}{gr(f' {ver_}')}"
            row += f"    {entry}"
            if (i + 1) % 3 == 0:
                print(row)
                row = ""
        if row:
            print(row)
        print()

    # 기타
    other = by_cat.get("🧰 유틸 / 기타", [])
    if other:
        other.sort(key=lambda x: x[0].lower())
        names = "  ".join(g(n) + gr(f" {v}") for n, v in other)
        print(f"  {b('🧰 유틸 / 기타')}  ({len(other)}개)")
        # 한 줄에 4개씩
        for i in range(0, len(other), 4):
            chunk = other[i:i+4]
            print("    " + "   ".join(g(n) + gr(f" {v}") for n, v in chunk))
        print()


# ═══════════════════════════════════════════════════════════════════
# 2. 프로젝트 .py 파일 분석
# ═══════════════════════════════════════════════════════════════════

def parse_file(filepath):
    info = {
        "imports": [], "from_imports": [],
        "functions": [], "classes": [], "top_vars": [],
        "first_comment": "", "lines": 0, "size_kb": 0.0, "error": None,
    }
    path = Path(filepath)
    info["size_kb"] = round(path.stat().st_size / 1024, 1)

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()
    lines = source.splitlines()
    info["lines"] = len(lines)

    for line in lines[:6]:
        s = line.strip()
        if s.startswith("#") and len(s) > 1:
            info["first_comment"] = s[1:].strip()
            break

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        info["error"] = str(e)
        return info

    # 모듈 독스트링
    if not info["first_comment"] and tree.body:
        first = tree.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            info["first_comment"] = str(first.value.s).strip().splitlines()[0][:80]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                info["imports"].append((a.name, a.asname))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = [a.name for a in node.names]
            info["from_imports"].append((mod, names))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            suffix = " (async)" if isinstance(node, ast.AsyncFunctionDef) else ""
            info["functions"].append(node.name + suffix)
        elif isinstance(node, ast.ClassDef):
            info["classes"].append(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    info["top_vars"].append(t.id)

    return info


def top_pkg(mod):
    return mod.split(".")[0] if mod else ""


def collect_libs(info):
    seen, libs = set(), []
    for mod, alias in info["imports"]:
        t = top_pkg(mod)
        if t and t not in seen:
            seen.add(t); libs.append(t)
    for mod, names in info["from_imports"]:
        t = top_pkg(mod)
        if t and t not in seen:
            seen.add(t); libs.append(t)
    return libs


def show_files(target_dir):
    py_files = sorted(
        p for p in Path(target_dir).glob("*.py")
        if p.name != Path(__file__).name
    )
    if not py_files:
        print(rd("Python 파일 없음"))
        return {}

    print(f"\n{cy('═'*70)}")
    print(f" {b('📂 프로젝트 파일 분석')}  —  {Path(target_dir).resolve()}")
    print(cy('═'*70))

    all_info = {}
    for idx, path in enumerate(py_files, 1):
        info = parse_file(path)
        all_info[path.name] = info

        print(f"\n {gr(f'[{idx}/{len(py_files)}]')} {b(path.name)}  {gr(f'{info[\"lines\"]}줄  {info[\"size_kb\"]}KB')}")

        if info["first_comment"]:
            print(f"   {gr('설명:')} {info['first_comment']}")

        if info["error"]:
            print(f"   {rd('⚠ 파싱 오류:')} {info['error']}")
            continue

        libs = collect_libs(info)
        if libs:
            lib_str = "  ".join(g(l) for l in libs)
            print(f"   {y('📦')} {lib_str}")

        if info["classes"]:
            print(f"   {m('🔷 클래스:')} " + "  ".join(m(c) for c in info["classes"]))

        if info["functions"]:
            fns = info["functions"]
            fn_str = "  ".join(bl(f) for f in fns[:8])
            extra = f"  {gr(f'+{len(fns)-8}개')}" if len(fns) > 8 else ""
            print(f"   {bl('🔧 함수:')} {fn_str}{extra}")

        if info["top_vars"]:
            print(f"   {y('📌 상수:')} " + "  ".join(y(v) for v in info["top_vars"][:6]))

    return all_info


def show_project_lib_summary(all_info):
    lib_files = defaultdict(list)
    for fname, info in all_info.items():
        for lib in collect_libs(info):
            lib_files[lib].append(Path(fname).stem)

    if not lib_files:
        return

    print(f"\n{cy('─'*70)}")
    print(f" {b('📚 프로젝트 라이브러리 사용 요약')}")
    print(cy('─'*70))

    by_cat = defaultdict(list)
    for lib, files in lib_files.items():
        cat = categorize(lib)
        by_cat[cat].append((lib, files))

    for cat in list(CATEGORIES.keys()):
        items = by_cat.get(cat, [])
        if not items:
            continue
        items.sort(key=lambda x: -len(x[1]))
        print(f"\n  {b(cat)}")
        for lib, files in items:
            bar = "█" * len(files)
            flist = ", ".join(files[:4]) + (f" +{len(files)-4}" if len(files) > 4 else "")
            print(f"    {g(lib):<22} {bl(bar):<12}  {gr(flist)}")

    other = by_cat.get("🧰 유틸 / 기타", [])
    if other:
        other.sort(key=lambda x: -len(x[1]))
        print(f"\n  {b('🧰 유틸 / 기타')}")
        for lib, files in other:
            bar = "█" * len(files)
            flist = ", ".join(files[:4]) + (f" +{len(files)-4}" if len(files) > 4 else "")
            print(f"    {g(lib):<22} {bl(bar):<12}  {gr(flist)}")

    total_lines = sum(i["lines"] for i in all_info.values())
    total_kb = sum(i["size_kb"] for i in all_info.values())
    print(f"\n  {b(len(all_info))}개 파일  {b(total_lines)}줄  {b(f'{total_kb:.1f}')}KB  라이브러리 {b(len(lib_files))}종 사용")
    print(cy('═'*70))


# ═══════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.system("")  # Windows ANSI 활성화

    target = sys.argv[1] if len(sys.argv) > 1 else "."

    show_env()
    all_info = show_files(target)
    if all_info:
        show_project_lib_summary(all_info)
