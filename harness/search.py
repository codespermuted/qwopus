"""Specialized search tools — GitHub, Google Scholar, and Stack Overflow scraping."""
from __future__ import annotations

import json
import logging
import re
import subprocess
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


# ── GitHub search (by stars) ─────────────────────────────────

def github_search(query: str, max_results: int = 5, language: str = "") -> str:
    """Search GitHub repositories sorted by stars."""
    q = query
    if language:
        q += f" language:{language}"

    try:
        result = subprocess.run(
            ["gh", "api", "search/repositories",
             "-X", "GET",
             "-f", f"q={q}",
             "-f", "sort=stars",
             "-f", "order=desc",
             "-f", f"per_page={max_results}"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return f"GitHub API error: {result.stderr[:200]}"

        data = json.loads(result.stdout)
    except FileNotFoundError:
        return "The gh CLI is not installed."
    except Exception as e:
        return f"GitHub search failed: {e}"

    items = data.get("items", [])
    if not items:
        return "No search results."

    output_parts = []
    for i, repo in enumerate(items, 1):
        name = repo.get("full_name", "")
        stars = repo.get("stargazers_count", 0)
        desc = repo.get("description", "") or ""
        url = repo.get("html_url", "")
        lang = repo.get("language", "") or ""
        updated = repo.get("updated_at", "")[:10]
        topics = ", ".join(repo.get("topics", [])[:5])

        parts = [f"[{i}] {name} * {stars:,}"]
        parts.append(f"    {url}")
        if desc:
            parts.append(f"    {desc[:120]}")
        meta = []
        if lang:
            meta.append(lang)
        if updated:
            meta.append(f"updated:{updated}")
        if topics:
            meta.append(f"topics: {topics}")
        if meta:
            parts.append(f"    ({', '.join(meta)})")
        output_parts.append("\n".join(parts))

    total = data.get("total_count", 0)
    header = f"GitHub search: top {len(items)} of {total:,} (by stars)\n"
    return header + "\n\n".join(output_parts)


# ── Google Scholar scraping ──────────────────────────────────

def scholar_search(query: str, max_results: int = 5, year_from: int = 0, exclude_survey: bool = False) -> str:
    """Search Google Scholar for papers. Includes citation counts."""
    import requests
    from bs4 import BeautifulSoup

    search_query = query
    if exclude_survey:
        search_query += " -survey -review -comprehensive"

    params = f"q={quote_plus(search_query)}&num={max_results + 5}&hl=en"
    if year_from:
        params += f"&as_ylo={year_from}"

    url = f"https://scholar.google.com/scholar?{params}"

    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Failed to reach Google Scholar: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    results = soup.select("div.gs_r.gs_or.gs_scl")

    if not results:
        # Blocked by CAPTCHA or similar
        if "captcha" in resp.text.lower() or "unusual traffic" in resp.text.lower():
            return "Blocked by CAPTCHA on Google Scholar. Try again later."
        return "No search results."

    output_parts = []
    for i, item in enumerate(results[:max_results], 1):
        # Title
        title_tag = item.select_one("h3.gs_rt a")
        if not title_tag:
            title_tag = item.select_one("h3.gs_rt")
        title = title_tag.get_text(strip=True) if title_tag else "(no title)"
        link = title_tag.get("href", "") if title_tag and title_tag.name == "a" else ""

        # Author/source info
        meta_tag = item.select_one("div.gs_a")
        meta = meta_tag.get_text(strip=True) if meta_tag else ""

        # Snippet
        snippet_tag = item.select_one("div.gs_rs")
        snippet = snippet_tag.get_text(strip=True)[:150] if snippet_tag else ""

        # Citation count
        cite_tag = item.select_one("div.gs_fl a")
        citations = ""
        for a_tag in item.select("div.gs_fl a"):
            text = a_tag.get_text()
            if "Cited by" in text:
                citations = text.replace("Cited by ", "").strip()
                break

        # PDF link
        pdf_tag = item.select_one("div.gs_ggs a")
        pdf_url = pdf_tag.get("href", "") if pdf_tag else ""

        parts = [f"[{i}] {title}"]
        if link:
            parts.append(f"    {link}")
        if meta:
            parts.append(f"    {meta}")
        if citations:
            parts.append(f"    Cited by {citations}")
        if pdf_url:
            parts.append(f"    PDF: {pdf_url}")
        if snippet:
            parts.append(f"    {snippet}")

        output_parts.append("\n".join(parts))

    return "\n\n".join(output_parts)


# ── Stack Overflow scraping ──────────────────────────────────

def stackoverflow_search(query: str, max_results: int = 5) -> str:
    """Search Stack Overflow sorted by votes via the API (no API key required)."""
    import requests

    try:
        resp = requests.get(
            "https://api.stackexchange.com/2.3/search/advanced",
            params={
                "order": "desc",
                "sort": "votes",
                "q": query,
                "site": "stackoverflow",
                "pagesize": max_results,
                "filter": "withbody",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Stack Overflow search failed: {e}"

    items = data.get("items", [])
    if not items:
        return "No search results."

    output_parts = []
    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        url = item.get("link", "")
        score = item.get("score", 0)
        answers = item.get("answer_count", 0)
        is_answered = item.get("is_answered", False)
        tags = ", ".join(item.get("tags", [])[:5])

        status = "answered" if is_answered else "unanswered"
        parts = [f"[{i}] {title}"]
        parts.append(f"    {url}")
        parts.append(f"    {score} votes * {answers} answers * {status}")
        if tags:
            parts.append(f"    tags: {tags}")

        output_parts.append("\n".join(parts))

    return "\n\n".join(output_parts)
