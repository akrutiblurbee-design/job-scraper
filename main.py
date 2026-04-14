from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import io
import os
import re

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from supabase import create_client, Client
from jobspy import scrape_jobs

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger


# ─── Config ───────────────────────────────────────────────────────────────────

PAST_DAYS = 7
HOURS_OLD = PAST_DAYS * 24
RESULTS_WANTED = 70
SITE = ["linkedin"]
JOB_TYPES = ["fulltime", "contract"]
INCLUDE_RANK_SCORE = True
MAX_RANK_SCORE = 10

# ─── Scheduler Config ─────────────────────────────────────────────────────────

# Set the time you want the scraper to run daily (IST = UTC+5:30)
SCRAPE_HOUR_IST   = 10     # 7 AM IST
SCRAPE_MINUTE_IST = 46   # 7:11 AM IST
IST = ZoneInfo("Asia/Kolkata")

# ─── Supabase Config ──────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = "Job-scraper"
S3_PREFIX = "jobs"
FILE_RETENTION_DAYS = 3

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ─── Keywords & Search Terms ──────────────────────────────────────────────────

RANK_KEYWORDS = {
    "python": 4,
    "django": 2,
    "fastapi": 3,
    "flask": 2,
    "llm": 3,
    "aiml": 3,
    "genai": 4,
    "machine learning": 3,
    "ai": 2,
    "tensorflow": 3,
    "pytorch": 3,
    "langchain": 3,
    "langgraph": 3,
    "ai agents": 3,
    "angular": 4,
    "node js": 2,
    "react": 3,
    "react native": 3,
    "android": 3,
    "ios": 3,
    "frontend": 4,
    "backend": 4,
    "full stack": 4,
}

WEB_DEV_TERMS = [
    "saas",
    "startup",
    "full stack developer",
    "mern stack developer",
    "frontend developer",
    "backend developer",
]

AI_TERMS = [
    "ai engineer",
    "machine learning engineer",
    "AI ML developer",
    "data scientist",
    "llm",
    "python",
    "tensorflow",
]

MOBILE_TERMS = [
    "react native developer",
    "mobile developer",
    "android developer",
]


# ─── Supabase Storage Helpers ─────────────────────────────────────────────────

import httpx

def save_csv_to_supabase(df: pd.DataFrame):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8")
    csv_bytes = buffer.getvalue().encode("utf-8")

    today = datetime.now().strftime("%Y-%m-%d")
    paths = [
        f"{S3_PREFIX}/{today}_jobs.csv",
        f"{S3_PREFIX}/latest.csv",
    ]

    base_url = SUPABASE_URL.rstrip("/")
    bucket = SUPABASE_BUCKET

    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
        "Content-Type": "text/csv",
        "x-upsert": "true",
    }

    for path in paths:
        url = f"{base_url}/storage/v1/object/{bucket}/{path}"
        print(f"  Uploading to: {url}")

        with httpx.Client() as client:
            response = client.post(url, content=csv_bytes, headers=headers, timeout=30)

        print(f"  Response: HTTP {response.status_code} — {response.text[:200]}")

        if response.status_code not in (200, 201):
            raise Exception(
                f"Failed to upload {path}: HTTP {response.status_code} — {response.text}"
            )
        print(f"  Saved -> {path}")


def read_latest_csv_from_supabase() -> pd.DataFrame:
    """Download jobs/latest.csv from Supabase Storage."""
    response = supabase.storage.from_(SUPABASE_BUCKET).download(
        f"{S3_PREFIX}/latest.csv"
    )
    return pd.read_csv(io.BytesIO(response))


def delete_old_files_from_supabase():
    """Delete dated CSV files older than FILE_RETENTION_DAYS. Never deletes latest.csv."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=FILE_RETENTION_DAYS)

    files = supabase.storage.from_(SUPABASE_BUCKET).list(S3_PREFIX)

    to_delete = []
    for f in files:
        if f["name"] == "latest.csv":
            continue

        updated_at_str = f.get("updated_at") or f.get("created_at", "")
        try:
            last_modified = datetime.fromisoformat(
                updated_at_str.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            continue

        if last_modified < cutoff:
            to_delete.append(f"{S3_PREFIX}/{f['name']}")

    if to_delete:
        supabase.storage.from_(SUPABASE_BUCKET).remove(to_delete)
        print(f"  Deleted {len(to_delete)} old file(s): {to_delete}")
    else:
        print("  No old files to delete.")


# ─── Core Logic ───────────────────────────────────────────────────────────────

def calculate_rank_score(
    title,
    description,
    keyword_weights: dict[str, int],
    max_rank_score: int = 10,
) -> tuple[int, str]:
    """Return a normalized keyword match score and matched keywords."""
    combined_text = f"{title or ''} {description or ''}".lower()

    if not combined_text.strip() or not keyword_weights:
        return 0, ""

    matched_keywords = []
    matched_weight = 0
    total_weight = sum(keyword_weights.values())

    for keyword, weight in keyword_weights.items():
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        if re.search(pattern, combined_text):
            matched_keywords.append(keyword)
            matched_weight += weight

    if total_weight <= 0:
        return 0, ", ".join(matched_keywords)

    normalized_score = round((matched_weight / total_weight) * max_rank_score)
    return min(normalized_score, max_rank_score), ", ".join(matched_keywords)


def extract_work_location(description):
    """Extract work location from job description text."""
    if not description or pd.isna(description):
        return "Unknown"

    desc = str(description).lower()

    if re.search(r"\bremote\b", desc):
        return "Remote"
    if re.search(r"\bhybrid\b", desc):
        return "Hybrid"
    if re.search(r"\bon-site\b|\bin-office\b|\bon site\b", desc):
        return "On-site"

    return "Unknown"


def scrape_category(search_terms: list[str], category_label: str) -> pd.DataFrame:
    all_frames = []

    for term in search_terms:
        for job_type in JOB_TYPES:
            print(f"  [{category_label}] Searching: '{term}' ({job_type}) ...")
            try:
                df = scrape_jobs(
                    site_name=SITE,
                    search_term=term,
                    hours_old=HOURS_OLD,
                    job_type=job_type,
                    linkedin_fetch_description=True,
                    description_format="markdown",
                    results_wanted=RESULTS_WANTED,
                )
                if not df.empty:
                    df["search_term"] = term
                    df["queried_job_type"] = job_type
                    all_frames.append(df)
            except Exception as e:
                print(f"    Error scraping '{term}' ({job_type}): {e}")

    if not all_frames:
        print(f"  [{category_label}] No results found.")
        return pd.DataFrame()

    valid_frames = []
    for df in all_frames:
        if df.empty:
            continue
        cleaned_df = df.dropna(axis=1, how="all")
        if cleaned_df.empty:
            continue
        valid_frames.append(cleaned_df)

    if not valid_frames:
        print(f"  [{category_label}] No usable results found.")
        return pd.DataFrame()

    combined = pd.concat(valid_frames, ignore_index=True)
    combined.drop_duplicates(subset=["job_url"], inplace=True)

    if "description" in combined.columns:
        combined["work_location"] = combined["description"].apply(extract_work_location)
    else:
        combined["work_location"] = "Unknown"

    if "date_posted" in combined.columns:
        cutoff = datetime.now(timezone.utc) - timedelta(days=PAST_DAYS)
        combined["date_posted_parsed"] = pd.to_datetime(
            combined["date_posted"], utc=True, errors="coerce"
        )
        combined = combined[
            combined["date_posted_parsed"].isna()
            | (combined["date_posted_parsed"] >= cutoff)
        ].copy()
    else:
        combined["date_posted_parsed"] = pd.NaT

    print(f"  [{category_label}] {len(combined)} jobs after date filter")

    remote_only = combined[combined["work_location"] == "Remote"].copy()
    print(f"  [{category_label}] {len(remote_only)} remote jobs before ranking")

    if remote_only.empty:
        remote_only["rank_score"] = pd.Series(dtype=int)
        remote_only["matched_keywords"] = pd.Series(dtype=str)
    else:
        remote_only[["rank_score", "matched_keywords"]] = remote_only.apply(
            lambda row: pd.Series(
                calculate_rank_score(
                    row.get("title", ""),
                    row.get("description", ""),
                    RANK_KEYWORDS,
                    MAX_RANK_SCORE,
                )
            ),
            axis=1,
        )

    remote_only = remote_only[remote_only["rank_score"] >= 1].copy()

    sort_columns = ["rank_score"]
    sort_order = [False]
    if "date_posted_parsed" in remote_only.columns:
        sort_columns.append("date_posted_parsed")
        sort_order.append(False)
    remote_only.sort_values(by=sort_columns, ascending=sort_order, inplace=True)

    print(f"  [{category_label}] {len(combined)} total -> {len(remote_only)} remote jobs kept.")

    cols = ["title", "company", "job_url", "work_location", "queried_job_type"]
    if INCLUDE_RANK_SCORE:
        cols.append("rank_score")
    available = [c for c in cols if c in remote_only.columns]
    result = remote_only[available].copy()
    result["category"] = category_label
    return result


def run_scraper() -> pd.DataFrame:
    """Run all scraping categories, save to Supabase Storage, clean up old files."""
    print(f"\n[{datetime.now()}] Starting scrape run...")

    print("\nScraping Web Dev jobs...")
    web_dev_jobs = scrape_category(WEB_DEV_TERMS, "Web Dev")

    print("\nScraping AI jobs...")
    ai_jobs = scrape_category(AI_TERMS, "AI")

    print("\nScraping Mobile App Dev jobs...")
    mobile_jobs = scrape_category(MOBILE_TERMS, "Mobile")

    all_jobs = pd.concat([web_dev_jobs, ai_jobs, mobile_jobs], ignore_index=True)

    if not all_jobs.empty and "rank_score" in all_jobs.columns:
        all_jobs.sort_values(by="rank_score", ascending=False, inplace=True)
    if not all_jobs.empty and "job_url" in all_jobs.columns:
        all_jobs.drop_duplicates(subset=["job_url"], inplace=True)
    if not all_jobs.empty and "rank_score" in all_jobs.columns:
        all_jobs.sort_values(by="rank_score", ascending=False, inplace=True)

    print("\nSaving to Supabase Storage...")
    save_csv_to_supabase(all_jobs)

    print("\nCleaning up old files (>3 days)...")
    delete_old_files_from_supabase()

    print("\n-----------------------------")
    print(f"Web Dev  : {len(web_dev_jobs)} remote jobs")
    print(f"AI       : {len(ai_jobs)} remote jobs")
    print(f"Mobile   : {len(mobile_jobs)} remote jobs")
    print(f"Combined : {len(all_jobs)} remote jobs")
    print("Done!")

    return all_jobs


# ─── Scheduled Job Wrapper ────────────────────────────────────────────────────

def scheduled_scrape():
    """Wrapper called by APScheduler. Runs once daily at the configured IST time."""
    print(f"\n[APScheduler] Triggered scheduled scrape at {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    try:
        run_scraper()
        print(f"[APScheduler] Scheduled scrape completed successfully.")
    except Exception as e:
        print(f"[APScheduler] Scheduled scrape FAILED: {e}")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Job Scraper API",
    description="Scrapes LinkedIn remote jobs and returns ranked results.",
    version="4.1.0",
)


# ─── Scheduler Startup / Shutdown ─────────────────────────────────────────────

@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler(timezone=IST)
    scheduler.add_job(
        func=scheduled_scrape,
        trigger=CronTrigger(
            hour=SCRAPE_HOUR_IST,
            minute=SCRAPE_MINUTE_IST,
            timezone=IST,
        ),
        id="daily_scrape",
        name="Daily LinkedIn Job Scrape",
        replace_existing=True,   # Safe to redeploy without duplicate jobs
        max_instances=1,         # Ensures it runs ONLY ONCE even if previous run is still going
        coalesce=True,           # If missed fires on recovery, run only once
    )
    scheduler.start()
    next_run = scheduler.get_job("daily_scrape").next_run_time
    print(f"[APScheduler] Scheduler started. Next scrape at: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Store scheduler on app state so we can shut it down cleanly
    app.state.scheduler = scheduler


@app.on_event("shutdown")
def stop_scheduler():
    scheduler = getattr(app.state, "scheduler", None)
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
        print("[APScheduler] Scheduler shut down.")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def health_check():
    """Health check endpoint. Also shows next scheduled scrape time."""
    scheduler = getattr(app.state, "scheduler", None)
    next_run = None
    if scheduler:
        job = scheduler.get_job("daily_scrape")
        if job and job.next_run_time:
            next_run = job.next_run_time.strftime("%Y-%m-%d %H:%M:%S %Z")

    return {
        "status": "ok",
        "message": "Job scraper API is running",
        "scheduler": "active" if scheduler and scheduler.running else "inactive",
        "next_scheduled_scrape": next_run,
    }


@app.get("/scheduler/status", summary="Check scheduler status")
def scheduler_status():
    """Returns scheduler info and next scheduled run time."""
    scheduler = getattr(app.state, "scheduler", None)
    if not scheduler:
        return {"running": False, "jobs": []}

    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.strftime("%Y-%m-%d %H:%M:%S %Z") if job.next_run_time else None,
        })

    return {
        "running": scheduler.running,
        "jobs": jobs,
    }


@app.post("/scrape", summary="Manually trigger scraper → return JSON")
def scrape_json():
    """Manually trigger a full scrape. Saves to Supabase Storage and cleans old files."""
    try:
        df = run_scraper()
        df = df.where(pd.notnull(df), None)
        jobs = df.to_dict(orient="records")
        return JSONResponse(content={
            "success": True,
            "count": len(jobs),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "jobs": jobs,
        })
    except Exception as e:
        print(f"  Scraper error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scrape/csv", summary="Manually trigger scraper → download CSV")
def scrape_csv():
    """
    Manually trigger a full scrape and return as downloadable CSV.
    WARNING: Takes ~15 min. Use /scrape/csv/cached for n8n.
    """
    try:
        df = run_scraper()
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, encoding="utf-8")
        buffer.seek(0)
        filename = f"jobs_ranked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        print(f"  Scraper error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scrape/csv/cached", summary="← USE THIS IN n8n — returns latest CSV instantly")
def scrape_csv_cached():
    """
    Returns jobs/latest.csv from Supabase Storage. Responds in milliseconds.
    Pre-populated every day at 7:11 AM IST by APScheduler.
    Set your n8n trigger to 8:00 AM IST (to give scraper time to finish).
    """
    try:
        df = read_latest_csv_from_supabase()
    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg.lower() or "404" in error_msg:
            raise HTTPException(
                status_code=404,
                detail="No cached CSV found yet. APScheduler runs at 7:11 AM IST. "
                       "Or trigger manually via POST /scrape.",
            )
        raise HTTPException(status_code=500, detail=f"Storage read error: {error_msg}")

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8")
    buffer.seek(0)
    filename = f"jobs_ranked_{datetime.now().strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/scrape/csv/list", summary="List all CSV files in Supabase Storage")
def list_csv_files():
    """List all files in the jobs/ folder — useful for debugging retention cleanup."""
    try:
        files = supabase.storage.from_(SUPABASE_BUCKET).list(S3_PREFIX)
        return JSONResponse(content={
            "count": len(files),
            "files": [
                {
                    "name": f["name"],
                    "updated_at": f.get("updated_at"),
                    "size_kb": round(f["metadata"]["size"] / 1024, 2)
                    if f.get("metadata") else None,
                }
                for f in files
            ],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
