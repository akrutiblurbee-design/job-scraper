from datetime import datetime, timedelta, timezone
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import json
import os
import re

import pandas as pd
from jobspy import scrape_jobs


# ─── Config ───────────────────────────────────────────────────────────────────

PAST_DAYS = 7
HOURS_OLD = PAST_DAYS * 24
RESULTS_WANTED = 50
SITE = ["linkedin"]
JOB_TYPES = ["fulltime", "contract"]
OUTPUT_PATH = "output/jobs_ranked.csv"
INCLUDE_RANK_SCORE = True
MAX_RANK_SCORE = 10

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
    """
    Scrape LinkedIn for each search term and each job type,
    combine results, deduplicate, keep only past-week postings,
    and keep remote only.
    """
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
            combined["date_posted_parsed"].notna()
            & (combined["date_posted_parsed"] >= cutoff)
        ].copy()
    else:
        combined = combined.iloc[0:0].copy()

    remote_only = combined[combined["work_location"] == "Remote"].copy()

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

    cols = [
        "title",
        "company",
        "job_url",
        "work_location",
        "queried_job_type",
    ]
    if INCLUDE_RANK_SCORE:
        cols.append("rank_score")
    available = [c for c in cols if c in remote_only.columns]
    result = remote_only[available].copy()
    result["category"] = category_label
    return result


def save_csv(df: pd.DataFrame, filepath: str):
    """Save DataFrame to CSV."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"  Saved {len(df)} jobs -> {output_path}")
    except PermissionError:
        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        fallback_path = output_path.with_name(
            f"{output_path.stem}_{timestamp}{output_path.suffix}"
        )
        df.to_csv(fallback_path, index=False, encoding="utf-8")
        print(f"  Saved {len(df)} jobs -> {fallback_path} (fallback)")


def run_scraper() -> list[dict]:
    """
    Run all scraping categories and return combined jobs as a list of dicts.
    Also saves a local CSV as backup.
    """
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

    # Save CSV backup locally
    save_csv(all_jobs, OUTPUT_PATH)

    print("\n-----------------------------")
    print(f"Web Dev  : {len(web_dev_jobs)} remote jobs")
    print(f"AI       : {len(ai_jobs)} remote jobs")
    print(f"Mobile   : {len(mobile_jobs)} remote jobs")
    print(f"Combined : {len(all_jobs)} remote jobs")
    print("Done!")

    # Convert to JSON-serializable list
    # Replace NaN with None so json.dumps doesn't fail
    all_jobs = all_jobs.where(pd.notnull(all_jobs), None)
    return all_jobs.to_dict(orient="records")


# ─── HTTP Server (for Railway / n8n trigger) ──────────────────────────────────

class ScraperHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        """Health check endpoint — Railway uses this to verify the app is running."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "message": "Job scraper is running"}).encode())

    def do_POST(self):
        """
        Trigger endpoint — n8n calls POST / to start a scrape.
        Runs the scraper synchronously and returns JSON results.
        """
        print(f"\n[{datetime.now()}] POST /  — scrape triggered via HTTP")

        try:
            jobs = run_scraper()
            response = {
                "success": True,
                "count": len(jobs),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "jobs": jobs,
            }
            status = 200
        except Exception as e:
            print(f"  Scraper error: {e}")
            response = {
                "success": False,
                "error": str(e),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "jobs": [],
            }
            status = 500

        body = json.dumps(response, default=str).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Use print so logs appear in Railway dashboard
        print(f"  [HTTP] {self.address_string()} - {format % args}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), ScraperHandler)
    print(f"[{datetime.now()}] Job Scraper HTTP server started on port {port}")
    print(f"  GET  /  → health check")
    print(f"  POST /  → trigger scrape (returns JSON)")
    server.serve_forever()
