import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config


def web_search(query: str) -> list[dict]:
    if not config.TAVILY_API_KEY or not config.WEB_SEARCH_ENABLED:
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        results = client.search(
            query=query,
            max_results=config.TAVILY_MAX_RESULTS,
            include_raw_content=True,
        )
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "source": "web",
            }
            for r in results.get("results", [])
        ]
    except Exception as e:
        print(f"[web_search] ошибка: {e}")
        return []
