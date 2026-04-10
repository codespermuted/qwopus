"""Web tools — search and page fetching."""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# If the CJK character ratio exceeds this, the result is filtered out (drop Chinese/Japanese results)
CJK_THRESHOLD = 0.3

# Low-quality domains (sites that tend to be search noise)
LOW_QUALITY_DOMAINS = {
    "zhihu.com", "baidu.com", "zhidao.baidu.com",
    "csdn.net", "jianshu.com", "163.com",
}


def web_search(query: str, max_results: int = 5) -> str:
    """Web search: DuckDuckGo -> filter -> return English/Korean results."""
    raw_results = _ddg_search(query, max_results * 3)

    if not raw_results:
        return "No search results."

    # Pass 1: filter by CJK ratio and low-quality domains
    filtered = _filter_results(raw_results, max_results)

    # Pass 2: if there aren't enough English results, retry against English-language sites
    if len(filtered) < max_results:
        site_query = query + " site:github.com OR site:stackoverflow.com OR site:medium.com OR site:dev.to"
        extra = _ddg_search(site_query, max_results)
        extra_filtered = _filter_results(extra, max_results - len(filtered))
        for r in extra_filtered:
            if r.get("href") not in {f.get("href") for f in filtered}:
                filtered.append(r)
            if len(filtered) >= max_results:
                break

    if not filtered:
        filtered = raw_results[:max_results]

    output_parts = []
    for i, r in enumerate(filtered, 1):
        title = r.get("title", "")
        url = r.get("href", "")
        snippet = r.get("body", "")
        output_parts.append(f"[{i}] {title}\n    {url}\n    {snippet}")

    return "\n\n".join(output_parts)


def _ddg_search(query: str, max_results: int) -> list[dict]:
    """Run a DuckDuckGo search."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return []
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(
                query, region="us-en", safesearch="moderate",
                backend="lite", max_results=max_results,
            ))
    except Exception:
        return []


def _filter_results(results: list[dict], limit: int) -> list[dict]:
    """Drop results with a high CJK ratio or from low-quality domains."""
    filtered = []
    for r in results:
        title = r.get("title", "")
        url = r.get("href", "")
        snippet = r.get("body", "")

        if any(domain in url for domain in LOW_QUALITY_DOMAINS):
            continue
        if _cjk_ratio(title + snippet) > CJK_THRESHOLD:
            continue

        filtered.append(r)
        if len(filtered) >= limit:
            break
    return filtered


def web_fetch(url: str) -> str:
    """Extract body text from a URL. HTML -> cleaned text."""
    import requests
    from bs4 import BeautifulSoup

    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
            },
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Failed to fetch page: {e}"

    content_type = resp.headers.get("content-type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        return f"Unsupported content type: {content_type}"

    soup = BeautifulSoup(resp.text, "html.parser")

    # Strip unwanted elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
        tag.decompose()

    # Extract the body (try article > main > body in order)
    main_content = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", {"role": "main"})
        or soup.find("div", {"id": "content"})
        or soup.body
    )

    if not main_content:
        return "Could not extract body text from the page."

    text = main_content.get_text(separator="\n", strip=True)

    # Clean up blank lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)

    # Cap the length (context savings)
    if len(text) > 4000:
        text = text[:3500] + "\n\n... (truncated, the full page is longer)"

    return text


def _cjk_ratio(text: str) -> float:
    """Compute the ratio of CJK (Chinese/Japanese/Han) characters in the text. Korean is excluded."""
    if not text:
        return 0.0
    cjk_count = 0
    total = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):  # Count letters only
            total += 1
            # CJK Unified Ideographs (Han) — Korean Hangul is excluded
            cp = ord(ch)
            if (0x4E00 <= cp <= 0x9FFF       # CJK Unified
                or 0x3400 <= cp <= 0x4DBF    # CJK Extension A
                or 0x3040 <= cp <= 0x309F    # Hiragana
                or 0x30A0 <= cp <= 0x30FF):  # Katakana
                cjk_count += 1
    return cjk_count / total if total > 0 else 0.0
