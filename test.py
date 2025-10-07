def web_search(query) -> str:
    """Search the web for current information and return results."""
    try:
        from exa_py import Exa

        exa = Exa(api_key="8e667dfa-c286-4915-993a-ba373995ffc7")

        result = exa.search_and_contents(
            query,
            text=True,
            type="keyword",
            num_results=3
        )
        
        # Format the results into a readable string
        if not result.results:
            return "No search results found for this query."
        
        formatted_results = []
        for idx, item in enumerate(result.results, 1):
            title = getattr(item, 'title', 'No title')
            text = getattr(item, 'text', '')
            url = getattr(item, 'url', '')
            
            # Truncate text to a reasonable length
            if text and len(text) > 500:
                text = text[:500] + "..."
            
            result_text = f"Result {idx}: {title}\n"
            if text:
                result_text += f"{text}\n"
            if url:
                result_text += f"Source: {url}"
            
            formatted_results.append(result_text)
        
        return "\n\n".join(formatted_results)
    
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Search error: {str(e)}. Please try rephrasing your query."


print(web_search("what time is it"))