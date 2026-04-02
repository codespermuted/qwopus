"""웹 도구 — 검색 및 페이지 크롤링."""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# CJK 문자 비율이 이 이상이면 필터링 (중국어/일본어 결과 제외)
CJK_THRESHOLD = 0.3

# 신뢰도 낮은 도메인 (검색 노이즈가 많은 사이트)
LOW_QUALITY_DOMAINS = {
    "zhihu.com", "baidu.com", "zhidao.baidu.com",
    "csdn.net", "jianshu.com", "163.com",
}


def web_search(query: str, max_results: int = 5) -> str:
    """웹 검색: DuckDuckGo → 필터링 → 영어/한국어 결과 반환."""
    raw_results = _ddg_search(query, max_results * 3)

    if not raw_results:
        return "검색 결과가 없습니다."

    # 1차: CJK 비율 + 저품질 도메인 필터링
    filtered = _filter_results(raw_results, max_results)

    # 2차: 영어 결과 부족 시 영어권 사이트로 재검색
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
    """DuckDuckGo 검색 실행."""
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
    """CJK 비율 높은 결과 + 저품질 도메인 제외."""
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
    """URL의 본문 텍스트를 추출한다. HTML → 정리된 텍스트."""
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
        return f"페이지 가져오기 실패: {e}"

    content_type = resp.headers.get("content-type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        return f"지원하지 않는 콘텐츠 타입: {content_type}"

    soup = BeautifulSoup(resp.text, "html.parser")

    # 불필요한 요소 제거
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
        tag.decompose()

    # 본문 추출 (article > main > body 순서로 시도)
    main_content = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", {"role": "main"})
        or soup.find("div", {"id": "content"})
        or soup.body
    )

    if not main_content:
        return "페이지에서 본문을 추출할 수 없습니다."

    text = main_content.get_text(separator="\n", strip=True)

    # 빈 줄 정리
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)

    # 길이 제한 (컨텍스트 절약)
    if len(text) > 4000:
        text = text[:3500] + "\n\n... (잘림, 전체 페이지가 더 김)"

    return text


def _cjk_ratio(text: str) -> float:
    """텍스트에서 CJK (중국어/일본어/한자) 문자의 비율을 계산한다. 한국어는 제외."""
    if not text:
        return 0.0
    cjk_count = 0
    total = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):  # 문자만 카운트
            total += 1
            # CJK Unified Ideographs (한자) — 한국어 한글은 제외
            cp = ord(ch)
            if (0x4E00 <= cp <= 0x9FFF       # CJK Unified
                or 0x3400 <= cp <= 0x4DBF    # CJK Extension A
                or 0x3040 <= cp <= 0x309F    # Hiragana
                or 0x30A0 <= cp <= 0x30FF):  # Katakana
                cjk_count += 1
    return cjk_count / total if total > 0 else 0.0
