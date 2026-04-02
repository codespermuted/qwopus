"""웹 도구 — 검색 및 페이지 크롤링."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> str:
    """DuckDuckGo로 검색하여 결과를 반환한다."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "duckduckgo-search 패키지가 필요합니다: pip install duckduckgo-search"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region="wt-wt", max_results=max_results))
    except Exception as e:
        return f"검색 실패: {e}"

    if not results:
        return "검색 결과가 없습니다."

    output_parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("href", "")
        snippet = r.get("body", "")
        output_parts.append(f"[{i}] {title}\n    {url}\n    {snippet}")

    return "\n\n".join(output_parts)


def web_fetch(url: str) -> str:
    """URL의 본문 텍스트를 추출한다. HTML → 정리된 텍스트."""
    import requests
    from bs4 import BeautifulSoup

    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (Qwopus Bot)"},
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
