
from googlesearch import search
import ollama
from typing import List, Generator

class GoogleQueryGenerator:
    """Class to generate optimized Google search queries from user input"""
    
    def __init__(self, model_name: str = "GoogleQuerySearcher"):
        self.model_name = model_name
    
    def generate_query(self, user_query: str) -> str:
        """Generate an optimized Google search query from user input"""
        prompt = f"User-query: {user_query}"
        response = ollama.chat(
            model=self.model_name, 
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']["content"]


class GoogleSearch:
    """Class to perform Google searches and retrieve URLs"""
    
    def __init__(self, num_results: int = 10):
        self.num_results = num_results
    
    def search(self, query: str) -> List[str]:
        """Perform Google search and return list of URLs"""
        try:
            search_results = search(query, num_results=self.num_results)
            return list(search_results)
        except Exception as e:
            print(f"Error during Google search: {e}")
            return []


class URLFilter:
    """Class to filter and rank URLs based on relevance to user query"""
    
    def __init__(self, model_name: str = "UrlFilterModel"):
        self.model_name = model_name
    
    def filter_urls(self, user_query: str, url_list: List[str]) -> List[str]:
        """Filter URLs based on relevance to the user query"""
        if not url_list:
            return []
            
        prompt = f"""
        user-query: {user_query}
        url-list: {url_list} 
        """
        
        try:
            response = ollama.chat(
                model=self.model_name, 
                messages=[{"role": "user", "content": prompt}]
            )
            url_indices = response['message']["content"].split(" ")
            
            # Filter and return the selected URLs
            filtered_urls = []
            for index in url_indices:
                try:
                    filtered_urls.append(url_list[int(index)])
                except (ValueError, IndexError):
                    continue
                    
            return filtered_urls
        except Exception as e:
            print(f"Error during URL filtering: {e}")
            return []


class WebUrlModule:
    """Main class to coordinate the URL search and filtering process"""
    
    def __init__(self, 
                 query_model: str = "GoogleQuerySearcher", 
                 filter_model: str = "UrlFilterModel",
                 num_results: int = 10):
        
        self.query_generator = GoogleQueryGenerator(query_model)
        self.search_engine = GoogleSearch(num_results)
        self.url_filter = URLFilter(filter_model)
    
    def process_query(self, user_query: str) -> List[str]:
        """Process a user query and return filtered URLs"""
        print(f"Processing query: {user_query}")
        
        # Generate optimized search query
        google_query = self.query_generator.generate_query(user_query)
        print(f"Generated search query: {google_query}")
        
        # Perform Google search
        google_links = self.search_engine.search(google_query)
        print(f"Found {len(google_links)} search results")
        
        # Filter URLs based on relevance
        filtered_urls = self.url_filter.filter_urls(user_query, google_links)
        print(f"Filtered to {len(filtered_urls)} relevant URLs")
        
        return filtered_urls


# Example usage
if __name__ == "__main__":
    web_module = WebUrlModule()
    
    query = input("User: ")
    results = web_module.process_query(query)
    
    print("\nRelevant URLs:")
    for i, url in enumerate(results, 1):
        print(f"{i}. {url}")