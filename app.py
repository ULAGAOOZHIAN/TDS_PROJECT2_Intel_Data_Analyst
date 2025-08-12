from fastapi import FastAPI
from dotenv import load_dotenv
# --- stdlib (built-in) ---
import os, re, io, sys, json, uuid, logging, subprocess, mimetypes, tempfile
from typing import Dict, Any, List, Tuple, Optional

# --- third-party (from requirements.txt) ---

from fastapi import UploadFile, File, Query, Request
from fastapi.responses import JSONResponse
from google import genai


app = FastAPI(title="Intelligent Data Analyst API")

@app.get("/")
def root():
    return {"message": "API running on Hugging Face!"}
# =============================
# Env & setup
# =============================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set.")

# os.makedirs(".matplotlib_tmp", exist_ok=True)
# os.makedirs(".duckdb_extensions", exist_ok=True)
# os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_tmp")
# os.environ.setdefault("DUCKDB_EXTENSION_DIRECTORY", ".duckdb_extensions")

# MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
# SUBPROCESS_TIMEOUT = int(os.getenv("SUBPROCESS_TIMEOUT", "240"))
# MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")

# os.makedirs("logs", exist_ok=True)
# logging.basicConfig(filename="logs/server_q2.log", level=logging.INFO,
#                   format="%(asctime)s [%(levelname)s] %(message)s")

# TMP_DIR = "/tmp"
# os.environ.setdefault("MPLCONFIGDIR", os.path.join(TMP_DIR, "mpl"))
# os.environ.setdefault("DUCKDB_EXTENSION_DIRECTORY", os.path.join(TMP_DIR, "duckdb_extensions"))
# os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
# os.makedirs(os.environ["DUCKDB_EXTENSION_DIRECTORY"], exist_ok=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
SUBPROCESS_TIMEOUT = int(os.getenv("SUBPROCESS_TIMEOUT", "240"))
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")

# Log to stdout (file logging can fail on HF)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

client = genai.Client(api_key=GEMINI_API_KEY)

# =============================
# Constraints (unchanged from your last paste)
# =============================
COMMON_REQUIREMENTS = r"""
You are a data scientist and Python expert.
You will receive a natural-language task (from FILE CONTENT) describing sources and specific analysis/plots.
Write a SINGLE raw Python script that:
- Imports only what’s available; DO NOT install packages at runtime. If a required package is missing, raise a clear error early: print a dict with an 'error' key explaining the missing dependency.
- For any plots: 1) export WebP under 100KB via base64 data URI. 2) Do NOT import seaborn or scipy; use only matplotlib + numpy for plotting/stats.
- If Matplotlib is used, assume MPLCONFIGDIR is already set; never write to ~/.matplotlib or .matplotlib_tmp
- If using DuckDB: run INSTALL/LOAD inside Python; set caching as allowed (enable_object_cache, enable_http_metadata_cache) — do NOT use s3_enable_object_cache.
- Never download or materialize large remote datasets; use server-side pushdown/filters and LIMITs where relevant.
- Normalize and reconcile column names/keys (lower/strip; regex/synonym mapping) to tolerate schema drift. Always check type before applying .lower() — apply only to strings (e.g., str(col).lower().strip()).
- Return ONLY a Python dict with exact keys requested by the question, plus optional 'provenance' or 'notes'.
- Assign the dict to a variable named result and print(json.dumps(result)) exactly — no markdown, no comments, no extra output.
- Keep the script fast and memory-efficient.
"""

REMOTE_QUERY_CONSTRAINTS = r"""
Important (remote_query):
- Do NOT download or materialize the full dataset. Query in-place.
- Prefer DuckDB with httpfs/parquet extensions.
- Use parquet_scan(path, hive_partitioning=1) with WHERE filters for partition/column pushdown.
- When converting strings to dates, use STRPTIME(str, '%d-%m-%Y') in DuckDB.
- Avoid str_to_date / try_str_to_date.
- For plots, cap rows via LIMIT (e.g., 10000) to avoid large transfers.
- Enable: INSTALL/LOAD httpfs & parquet; SET s3_region if needed; SET enable_object_cache=true (do NOT use s3_enable_object_cache).
- When filling missing values in Pandas, never use None — use np.nan or domain-appropriate default (0 for numeric, empty string for text), or a fill method like ffill/bfill.
"""

SCRAPE_CONSTRAINTS = r"""
Important (scrape):
- Fetch only minimal pages with requests (desktop User-Agent, timeout=(5,30)); small delays if multiple pages.
- Prefer pandas.read_html(url, flavor=['lxml']); else wrap response.text with io.StringIO() using correct encoding.
- Strip <!-- --> if tables are inside HTML comments. Flatten multi-row headers.
- Normalize headers; drop footers; clean footnote markers; coerce numerics/dates.
- Reconcile column names using fuzzy/regex mapping as also their semantic meaning vis-à-vis those referred to in data analysis questions.
- Identify correct tables by caption/heading context and semantic role.
- If not found, progressively relax rules (fuzzy → partial → caption + few cols); prefer most complete table.
- Harmonize schema post-extraction: create role→column mapping (e.g., year, rank, entity, measure).
- Avoid Selenium; process in-memory only.
- Retry on transient HTTP errors. Handle encodings.
- Limit rows before heavy processing (e.g., LIMIT 10000).
- If no matching table is found:
  - 1. Strip comment blocks and retry.
  - 2. Extract all tables and rank them by fuzzy match to expected columns.
  - 3. Return a detailed schema_report indicating why fallback was used and which table was chosen.
"""

CSV_EXCEL_CONSTRAINTS = r"""
Important (csv_excel):
- Stream/download only needed columns/rows; use pandas.read_csv(..., usecols=..., nrows=...).
- For remote files, use DuckDB read_csv_auto/parquet_scan with WHERE filters.
- Normalize headers; reconcile synonyms; set dtypes; handle encodings (utf-8, latin-1).
- Drop footer/meta rows; coerce numerics. Avoid saving to disk.
- Cap rows before plotting (e.g., LIMIT 10000).
- Always inspect and normalize df.columns before accessing expected fields; if expected columns are not found, include a schema_report in result and attempt to match with fuzzy or partial string matching.
- Use pandas.read_excel(..., engine='openpyxl') explicitly to surface engine errors clearly.
"""

PDF_TABLE_CONSTRAINTS = r"""
Important (pdf_table):
- Prefer pdfplumber or camelot/tabula (if installed); fallback to regex if structured tables are not found.
- Identify correct tables based on captions/headings and expected columns like 'title', 'subject', 'year'.
- Coalesce multi-line header rows into single flattened row using join/strip on non-None values.
- If title rows are fragmented across lines, concatenate adjacent line fragments when appropriate.
- Normalize headers to lowercase strings; drop footers or repeated header rows.
- Validate table width/shape; drop totals.
- If no valid table is found, return schema_report with sample raw text and inferred structure.
"""

XML_JSON_API_CONSTRAINTS = r"""
Important (xml_json_api):
- Use requests with timeout + small retry loop.
- XML: parse via ElementTree/lxml; robust XPath; normalize keys; handle pagination.
- JSON: json_normalize or manual flatten; normalize keys.
- Validate non-empty records. Produce concise schema_report on mismatch.
"""

MEDIA_TRANSCRIBE_CONSTRAINTS = r"""
Important (media_transcribe):
- Work only on provided segments/URLs (no huge downloads).
- If transcription libs missing, fail clearly.
- After transcription, do the requested analysis; state assumptions.
"""

GEO_DATA_CONSTRAINTS = r"""
Important (geo_data):
- DuckDB+SPATIAL or GeoPandas; reproject for distance/area; keep plots <100KB WebP.
"""

RETRY_HINTS = {
    "pdf_table": (
        "\n\nRetry Strategy Hint:\n"
        "- If structured table extraction fails (e.g., using pdfplumber), fallback to regex-based line parsing.\n"
        "- Look for patterns like 'Title Author Year Subject' in text lines.\n"
        "- Extract fields manually. If partial parsing succeeds, return a schema_report explaining the method used.\n"
    ),
    "scrape": (
        "\n\nRetry Strategy Hint:\n"
        "- If read_html fails or returns no tables, strip HTML comments (<!-- -->), or use BeautifulSoup + fuzzy matching of headers.\n"
        "- Use header similarity ≥ 0.6 to match expected roles like rank, title, film, gross, year.\n"
        "- Include schema_report describing table selection logic.\n"
    ),
    # You can expand this dictionary for other approaches (csv_excel, xml_json_api, etc.)
}

APPROACH_TO_CONSTRAINTS = {
    "scrape": SCRAPE_CONSTRAINTS,
    "remote_query": REMOTE_QUERY_CONSTRAINTS,
    "csv_excel": CSV_EXCEL_CONSTRAINTS,
    "pdf_table": PDF_TABLE_CONSTRAINTS,
    "xml_json_api": XML_JSON_API_CONSTRAINTS,
    "media_transcribe": MEDIA_TRANSCRIBE_CONSTRAINTS,
    "geo_data": GEO_DATA_CONSTRAINTS,
}

# =============================
# Prompt templates
# =============================
CLASSIFY_PROMPT_TMPL = """
You are a senior data scientist. Read the question and attachment manifest and decide the best approach.

Options:
- scrape (HTML/web tables)
- remote_query (S3 parquet/CSV/SQL to be queried in-place)
- csv_excel (CSV/Excel files)
- pdf_table (PDF tables or structured text)
- xml_json_api (XML/JSON API feeds)
- media_transcribe (audio/video to text then analyze)
- geo_data (geospatial formats)

Return STRICT JSON ONLY:
{{
  "approach": "<one of above>",
  "confidence": 0..1,
  "evidence": ["short reasons"],
  "notes": "short comment",
  "output_contract": "<if the question specifies an exact JSON shape, paste it verbatim here; else empty>"
}}

QUESTION:
{qtext}

ATTACHMENTS (local paths will be provided at runtime):
{manifest}
"""

CODEGEN_PROMPT_TMPL = """
You will now write a SINGLE Python script that performs the full task.

{common}
{approach_constraints}

Additional runtime context:
- All attachments are saved locally; use their paths from the manifest below.
- If output_contract (below) is non-empty, your result dict MUST strictly adhere to it (keys/types). If a key can't be produced, set a sensible null/empty value but keep the shape.
- For small text attachments (csv/json/txt) a short preview may be provided to help schema inference.

OUTPUT CONTRACT (may be empty):
{output_contract}

- Return ONLY a Python value that matches the OUTPUT CONTRACT exactly (object or array as specified).
- Do NOT include any additional keys such as: 'provenance', 'notes', 'schema_report', 'stderr', 'approach', 'analysis' — unless they are explicitly present in the OUTPUT CONTRACT.
- Assign it to result and print(json.dumps(result, ensure_ascii=False)).
- No markdown, no comments, no prints other than the dict string.

**Error-handling requirements (MANDATORY):**
- If any exception occurs OR you determine the output would violate the contract or include placeholder/diagnostic text (e.g., values starting with "Error:"), then do NOT print any dict to STDOUT.
- Instead, write a traceback to STDERR and exit with a non-zero status:

ATTACHMENT MANIFEST:
{manifest}

FILE CONTENT (questions.txt):
{qtext}

Return ONLY raw executable Python. No markdown, no comments.
"""

RETRY_PROMPT_TMPL = """
Fix the following Python script based on this error. Keep the same constraints and output contract.
Return ONLY raw executable Python.

Original Questions:
{qtext}

Chosen Approach: {approach}

Output Contract:
{output_contract}

Attachment Manifest:
{manifest}

Previous Script:
{script}

Error:
{stderr}
"""

# =============================
# Helpers
# =============================
def _looks_like_error_payload(payload) -> str | None:
    """
    Return an error message string if the payload indicates failure;
    return None if it looks like a successful payload.
    """
    # 1) Common pattern: strings starting with "Error:"
    def _has_error_text(val):
        return isinstance(val, str) and val.strip().lower().startswith("error:")

    # 2) Empty/invalid image data URIs
    def _is_empty_data_uri(val):
        return isinstance(val, str) and val.strip().startswith("data:image/") and len(val.strip()) <= 22  # e.g. "data:image/png;base64,"

    if isinstance(payload, dict):
        # Special-case contract wrapper
        if payload.get("status") == "error":
            return str(payload.get("message") or "Unknown error")

        # Any value an error string?
        for k, v in payload.items():
            if _has_error_text(v) or _is_empty_data_uri(v):
                return f"{k}: {v if isinstance(v, str) else 'invalid image data'}"

        # Also consider empty essentials (tune to your contracts)
        # e.g., required keys with empty values may signal a failure
        # if "answers" in payload and not payload["answers"]:
        #     return "Empty answers list"

    elif isinstance(payload, list):
        for v in payload:
            if _has_error_text(v) or _is_empty_data_uri(v):
                return str(v)

    return None

# ADD THIS helper (verbatim copy of your current /solve body, minus the decorator/signature)
def _solve_core(files: List[UploadFile], debug: bool = False):
    session_id = uuid.uuid4().hex[:8]

    # 1) Save uploads, get qtext + attachments manifest
    try:
        workdir, qtext, manifest = save_attachments(files, session_id)
    except Exception as e:
        return JSONResponse({"error": f"Upload processing failed: {e}"}, status_code=400)

    # 2) Extract explicit output contract from qtext
    explicit_contract = extract_output_contract(qtext)

    # 3) Classify with attachments
    analysis = classify_with_attachments(qtext, manifest)
    approach = analysis.get("approach", "scrape")
    output_contract = analysis.get("output_contract") or explicit_contract

    logging.info("[%s] approach=%s conf=%s", session_id, approach, analysis.get("confidence"))

    # 4) Build initial codegen prompt
    current_prompt = build_codegen_prompt(qtext, approach, output_contract, manifest)

    last_stderr = ""
    for attempt in range(MAX_RETRIES):
        resp = client.models.generate_content(model=MODEL_NAME, contents=current_prompt)
        script = clean_code_output((resp.text or "").strip())

        stdout, stderr, rc = run_script(script)
        logging.info("[%s] attempt=%d rc=%s stderr=%s", session_id, attempt + 1, rc, (stderr or "")[:500])

        # 5) Return ONLY the script's JSON (contract dictates shape)
        try:
            payload = parse_possible_json(stdout)
            if isinstance(payload, dict) and set(payload.keys()) == {"result"}:
                payload = payload["result"]
            payload = apply_contract_filter(payload, output_contract)

            # NEW: treat semantic error payloads as failure → retry
            err_msg = _looks_like_error_payload(payload)
            if err_msg:
                last_stderr = f"Semantic error payload detected: {err_msg}"
                # Fall through to build a retry prompt (do NOT return yet)
            else:
                if debug:
                    from fastapi import Response
                    return Response(content=json.dumps(payload), media_type="application/json")
                return JSONResponse(payload, status_code=200)

        except Exception as parse_err:
            last_stderr = (
                f"JSON parse error: {parse_err}; STDOUT: {stdout[:400]}..."
                if rc == 0 else (stderr or "Unknown error")
            )      
        # try:
        #    payload = parse_possible_json(stdout)
        #    if isinstance(payload, dict) and set(payload.keys()) == {"result"}:
        #        payload = payload["result"]
        #    payload = apply_contract_filter(payload, output_contract)

        #   if debug:
        #        from fastapi import Response
        #        return Response(content=json.dumps(payload), media_type="application/json")

        #    return JSONResponse(payload, status_code=200)

        # except Exception as parse_err:
        #    last_stderr = (
        #        f"JSON parse error: {parse_err}; STDOUT: {stdout[:400]}..."
        #        if rc == 0 else (stderr or "Unknown error")
        #    )

        # 6) Retry prompt — keep OUTPUT CONTRACT every time
        retry_hint = RETRY_HINTS.get(approach, "")
        manifest_lines = "\n".join([f"- {m['filename']} @ {m['path']}" for m in (manifest or [])]) or "(none)"
        current_prompt = (
            "Fix the following Python script based on this error. Keep the SAME constraints and output contract. "
            "Return ONLY raw executable Python."
            f"{retry_hint}\n\n"
            "OUTPUT CONTRACT (MANDATORY):\n"
            "- Return a JSON object with EXACTLY the keys from the Output Contract, no extras.\n"
            f"{output_contract or '(none provided; output must still be valid JSON matching the question)'}\n\n"
            f"Original Questions:\n{qtext}\n\n"
            f"Chosen Approach: {approach}\n\n"
            f"Attachments:\n{manifest_lines}\n\n"
            f"Previous Script:\n{script}\n\n"
            f"Error:\n{last_stderr}\n"
        )

    logging.error("[%s] All retries failed. Last error: %s", session_id, last_stderr)
    return JSONResponse({"error": "All retries failed"}, status_code=500)

def save_attachments(files: List[UploadFile], session_id: str) -> Tuple[str, str, List[dict]]:
    """
    Returns (workdir, qtext, manifest)
    - qtext: contents of the single questions.txt (first .txt encountered)
    - manifest: [{filename, mime, size, path, preview}]
    """
    workdir = tempfile.mkdtemp(prefix=f"job_{session_id}_")
    manifest = []
    qtext = ""

    for uf in files:
        data = uf.file.read()
        fpath = os.path.join(workdir, uf.filename)
        with open(fpath, "wb") as f:
            f.write(data)
        mime = uf.content_type or mimetypes.guess_type(uf.filename)[0] or "application/octet-stream"
        preview = ""
        try:
            if mime.startswith("text/") or uf.filename.lower().endswith((".txt", ".csv", ".json")):
                preview = data[:2000].decode("utf-8", errors="ignore")
        except Exception:
            preview = ""
        manifest.append({
            "filename": uf.filename,
            "mime": mime,
            "size": len(data),
            "path": fpath,
            "preview": preview
        })
        if not qtext and uf.filename.lower().endswith(".txt"):
            # pick the first .txt as the questions file
            try:
                qtext = data.decode("utf-8").strip()
            except Exception:
                qtext = ""

    if not qtext:
        raise ValueError("questions.txt not found in upload")

    return workdir, qtext, manifest


def extract_output_contract(qtext: str) -> str:
    """
    Try to capture an explicit 'Output format' / 'Output:' / JSON block from the question.
    If found, return the block verbatim; else empty.
    """
    # fenced json block
    m = re.search(r"```json\s*(\{.*?}|\[.*?])\s*```", qtext, flags=re.S|re.I)
    if m: return m.group(1).strip()
    # 'Output format:' until end
    m = re.search(r"(?is)(output\s*(format)?|return\s*shape|respond\s*with)\s*:\s*(\{.*)", qtext)
    if m: return m.group(3).strip()
    return ""

def apply_contract_filter(payload, contract_str: str):
    """
    Enforce the top-level shape from the output contract:
      - If contract is a dict => return ONLY those keys, in that order.
        If any required key is missing, raise ValueError (to trigger retry).
      - If contract is a list => require payload is a list of the same length.
      - If no/invalid contract => return payload unchanged.
    """
    if not contract_str:
        return payload

    try:
        tmpl = json.loads(contract_str)
    except Exception:
        return payload

    # Dict contract → key filtering + validation
    if isinstance(tmpl, dict):
        if not isinstance(payload, dict):
            raise ValueError("Result must be a JSON object with keys exactly as in the Output Contract.")
        keys = list(tmpl.keys())
        missing = [k for k in keys if k not in payload]
        if missing:
            raise ValueError(f"Missing required keys in result: {missing}. Do NOT add extra keys; output must be exactly {keys}.")
        # keep only the contract keys and in the same order
        return {k: payload[k] for k in keys}

    # List contract → length validation
    if isinstance(tmpl, list):
        if not isinstance(payload, list) or len(payload) != len(tmpl):
            raise ValueError("Result must be a JSON array with the same length as the Output Contract.")
        return payload

    return payload

def clean_code_output(raw: str) -> str:
    lines = raw.splitlines()
    out, started = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("```") or s.lower().startswith("python"):
            continue
        if not started:
            if s.startswith(("import","from","def","class")):
                started = True
            else:
                continue
        out.append(ln)
    return "\n".join(out).strip()

def parse_possible_json(s: str) -> Any:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b != -1 and b > a:
            return json.loads(s[a:b+1])
        raise

def run_script(code: str, timeout_sec: int = SUBPROCESS_TIMEOUT):
    import uuid, os, sys, subprocess, pathlib, tempfile

    sid = uuid.uuid4().hex[:8]
    spath = os.path.join(tempfile.gettempdir(), f"script_{sid}.py")  # e.g. /tmp/script_XXXX.py

    # all child writable dirs
    mpl_dir = "/tmp/mpl"
    duckdb_ext_dir = "/tmp/duckdb_extensions"
    home_dir = "/tmp"
    duckdb_home_dir = "/tmp/.duckdb"  # where DuckDB likes to keep some state

    # Ensure they exist from the parent too
    for d in (mpl_dir, duckdb_ext_dir, duckdb_home_dir):
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)

    # Bootstrap runs inside the child before any imports
    bootstrap = f"""\
import os, pathlib
os.environ["HOME"] = {home_dir!r}
os.environ["XDG_CACHE_HOME"] = {home_dir!r}
os.environ["MPLCONFIGDIR"] = {mpl_dir!r}
os.environ["DUCKDB_EXTENSION_DIRECTORY"] = {duckdb_ext_dir!r}
# make sure ~/.duckdb exists in this HOME
pathlib.Path({duckdb_home_dir!r}).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.environ["DUCKDB_EXTENSION_DIRECTORY"]).mkdir(parents=True, exist_ok=True)
"""

    with open(spath, "w", encoding="utf-8") as f:
        f.write(bootstrap)
        f.write("\n")
        f.write(code)

    try:
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = env.get("PYTHONWARNINGS", "ignore")
        # Belt-and-suspenders: set the same in the child env
        env["HOME"] = home_dir
        env["XDG_CACHE_HOME"] = home_dir
        env["MPLCONFIGDIR"] = mpl_dir
        env["DUCKDB_EXTENSION_DIRECTORY"] = duckdb_ext_dir

        result = subprocess.run(
            [sys.executable, "-W", "ignore", spath],
            capture_output=True,
            timeout=timeout_sec,
            env=env,
            text=True,
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1


# def run_script(code: str, timeout_sec: int = SUBPROCESS_TIMEOUT):
#    import tempfile, subprocess, uuid, os, sys

#    sid = uuid.uuid4().hex[:8]
#    spath = os.path.join(tempfile.gettempdir(), f"script_{sid}.py")  # e.g., /tmp/script_XXXX.py
#    with open(spath, "w", encoding="utf-8") as f:
#        f.write(code)
#
#    try:
#        env = os.environ.copy()
#        env["PYTHONWARNINGS"] = env.get("PYTHONWARNINGS", "ignore")
#        result = subprocess.run(
#            [sys.executable, "-W", "ignore", spath],
#            capture_output=True,
#            timeout=timeout_sec,
#            env=env,
#            text=True,
#        )
#        return result.stdout, result.stderr, result.returncode
#    except Exception as e:
#        return "", str(e), 1

def classify_with_attachments(qtext: str, manifest: list[dict]) -> Dict[str, Any]:
    man_txt = "\n".join(
        f"- {m['filename']} ({m['mime']}, {m['size']} bytes) @ {m['path']}"
        for m in (manifest or [])
    ) or "(none)"
    prompt = CLASSIFY_PROMPT_TMPL.format(qtext=qtext or "", manifest=man_txt)
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        info = parse_possible_json((resp.text or "").strip())
        if not isinstance(info, dict) or "approach" not in info:
            raise ValueError("classifier returned invalid payload")
        return info
    except Exception as e:
        logging.warning("Classifier failed (%s). Heuristic fallback.", e)
        # Heuristics by attachment extensions
        exts = {os.path.splitext(m.get("filename",""))[1].lower() for m in (manifest or [])}
        if {".csv", ".xlsx"}.intersection(exts):
            return {"approach":"csv_excel","confidence":0.6,"evidence":["ext"],"notes":"fallback","output_contract":""}
        if ".pdf" in exts:
            return {"approach":"pdf_table","confidence":0.6,"evidence":["ext"],"notes":"fallback","output_contract":""}
        if any(x in (qtext or "").lower() for x in ["s3://", "parquet", "duckdb"]):
            return {"approach":"remote_query","confidence":0.6,"evidence":["qtext"],"notes":"fallback","output_contract":""}
        return {"approach":"scrape","confidence":0.5,"evidence":["default"],"notes":"fallback","output_contract":""}

def build_codegen_prompt(qtext: str, approach: str, output_contract: str | None, manifest: list[dict]) -> str:
    constraints = APPROACH_TO_CONSTRAINTS.get(approach, SCRAPE_CONSTRAINTS)
    man_txt = "\n".join(
        f"- {m['filename']} ({m['mime']}, {m['size']} bytes) @ {m['path']}"
        + (f"\n  PREVIEW:\n  {m.get('preview')}" if m.get('preview') else "")
        for m in (manifest or [])
    ) or "(none)"
    contract_block = (
        "OUTPUT CONTRACT (MANDATORY):\n"
        + (output_contract or "No explicit block provided; return a valid JSON value answering the questions.")
        + "\n\nHARD REQUIREMENTS:\n"
        "- Print ONLY a single JSON object/array that EXACTLY matches the output contract above.\n"
        "- DO NOT include extra keys like 'provenance', 'notes', 'schema_report' unless the contract shows them.\n"
    )
    return CODEGEN_PROMPT_TMPL.format(
        common=COMMON_REQUIREMENTS,
        approach_constraints=constraints,
        output_contract=(output_contract or "(empty — just return the JSON exactly as requested in the question)"),
        manifest=man_txt,
        qtext=qtext or "",
    )


# =============================
# API
# =============================
@app.post("/api/")
async def school_api(
    questions: UploadFile = File(..., alias="questions.txt"),
    image: Optional[UploadFile] = File(None, alias="image.png"),
    data:  Optional[UploadFile] = File(None, alias="data.csv"),
    debug: bool = Query(False),
):
    files: List[UploadFile] = [questions] + [f for f in (image, data) if f]
    return _solve_core(files, debug)


@app.post("/solve")
def solve(files: List[UploadFile] = File(...), debug: bool = Query(False)):
    return _solve_core(files, debug)

