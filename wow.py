from googlesearch import search

def test_search():
    query = "OpenAI ChatGPT"
    # Use `num_results` or `stop` parameter depending on version
    for url in search(query, num_results=5, pause=2.0):
        print("→", url)

if __name__ == "__main__":
    test_search()
