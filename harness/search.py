"""특화 검색 도구 — GitHub, Google Scholar, Stack Overflow 크롤링."""
from __future__ import annotations

import json
import logging
import re
import subprocess
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


# ── GitHub 검색 (Star 순) ─────────────────────────────────────

def github_search(query: str, max_results: int = 5, language: str = "") -> str:
    """GitHub 레포를 Star 순으로 검색한다."""
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
            return f"GitHub API 오류: {result.stderr[:200]}"

        data = json.loads(result.stdout)
    except FileNotFoundError:
        return "gh CLI가 설치되어 있지 않습니다."
    except Exception as e:
        return f"GitHub 검색 실패: {e}"

    items = data.get("items", [])
    if not items:
        return "검색 결과가 없습니다."

    output_parts = []
    for i, repo in enumerate(items, 1):
        name = repo.get("full_name", "")
        stars = repo.get("stargazers_count", 0)
        desc = repo.get("description", "") or ""
        url = repo.get("html_url", "")
        lang = repo.get("language", "") or ""
        updated = repo.get("updated_at", "")[:10]
        topics = ", ".join(repo.get("topics", [])[:5])

        parts = [f"[{i}] {name} ⭐ {stars:,}"]
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
    header = f"GitHub 검색: {total:,}개 중 상위 {len(items)}개 (⭐ 순)\n"
    return header + "\n\n".join(output_parts)


# ── Google Scholar 크롤링 ─────────────────────────────────────

def scholar_search(query: str, max_results: int = 5, year_from: int = 0) -> str:
    """Google Scholar에서 논문을 검색한다. Citation 수 포함."""
    import requests
    from bs4 import BeautifulSoup

    params = f"q={quote_plus(query)}&num={max_results}&hl=en"
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
        return f"Google Scholar 접근 실패: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    results = soup.select("div.gs_r.gs_or.gs_scl")

    if not results:
        # CAPTCHA 등으로 차단된 경우
        if "captcha" in resp.text.lower() or "unusual traffic" in resp.text.lower():
            return "Google Scholar에서 CAPTCHA 차단됨. 잠시 후 다시 시도하세요."
        return "검색 결과가 없습니다."

    output_parts = []
    for i, item in enumerate(results[:max_results], 1):
        # 제목
        title_tag = item.select_one("h3.gs_rt a")
        if not title_tag:
            title_tag = item.select_one("h3.gs_rt")
        title = title_tag.get_text(strip=True) if title_tag else "(제목 없음)"
        link = title_tag.get("href", "") if title_tag and title_tag.name == "a" else ""

        # 저자/출처 정보
        meta_tag = item.select_one("div.gs_a")
        meta = meta_tag.get_text(strip=True) if meta_tag else ""

        # 스니펫
        snippet_tag = item.select_one("div.gs_rs")
        snippet = snippet_tag.get_text(strip=True)[:150] if snippet_tag else ""

        # Citation 수
        cite_tag = item.select_one("div.gs_fl a")
        citations = ""
        for a_tag in item.select("div.gs_fl a"):
            text = a_tag.get_text()
            if "Cited by" in text:
                citations = text.replace("Cited by ", "").strip()
                break

        # PDF 링크
        pdf_tag = item.select_one("div.gs_ggs a")
        pdf_url = pdf_tag.get("href", "") if pdf_tag else ""

        parts = [f"[{i}] {title}"]
        if link:
            parts.append(f"    {link}")
        if meta:
            parts.append(f"    {meta}")
        if citations:
            parts.append(f"    📊 Cited by {citations}")
        if pdf_url:
            parts.append(f"    📄 PDF: {pdf_url}")
        if snippet:
            parts.append(f"    {snippet}")

        output_parts.append("\n".join(parts))

    return "\n\n".join(output_parts)


# ── Stack Overflow 크롤링 ─────────────────────────────────────

def stackoverflow_search(query: str, max_results: int = 5) -> str:
    """Stack Overflow API로 투표 순 검색한다. (API 키 불필요)"""
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
        return f"Stack Overflow 검색 실패: {e}"

    items = data.get("items", [])
    if not items:
        return "검색 결과가 없습니다."

    output_parts = []
    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        url = item.get("link", "")
        score = item.get("score", 0)
        answers = item.get("answer_count", 0)
        is_answered = item.get("is_answered", False)
        tags = ", ".join(item.get("tags", [])[:5])

        status = "✅ 답변됨" if is_answered else "❓ 미답변"
        parts = [f"[{i}] {title}"]
        parts.append(f"    {url}")
        parts.append(f"    👍 {score} votes · 💬 {answers} answers · {status}")
        if tags:
            parts.append(f"    tags: {tags}")

        output_parts.append("\n".join(parts))

    return "\n\n".join(output_parts)
