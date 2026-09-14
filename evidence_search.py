import requests


def search_evidence(claim, max_results=5):
    url = "https://api.duckduckgo.com/"

    params = {
        "q": claim,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        results = []

        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", "DuckDuckGo result"),
                "url": data.get("AbstractURL", ""),
                "snippet": data.get("AbstractText", "")
            })

        for topic in data.get("RelatedTopics", []):
            if "Text" in topic and "FirstURL" in topic:
                results.append({
                    "title": topic["Text"][:100],
                    "url": topic["FirstURL"],
                    "snippet": topic["Text"]
                })

            if len(results) >= max_results:
                break

        return {
            "success": True,
            "message": "Search completed successfully.",
            "results": results[:max_results]
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "results": []
        }