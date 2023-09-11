import random
import requests
import time

# Function to fetch a random image from Unsplash
def fetch_random_image():
    # Replace 'YOUR_UNSPLASH_API_KEY' with your actual Unsplash API key
    api_key = 'rRYCLmIioBo7hoBpbsJjhyRpGHWYDHdQQPdkpLEwJ0c'
    
    # Define a list of keywords to use for random image searches
    keywords = [
        "nature",
        "animals",
        "city",
        "technology",
        "food",
        "art",
        "architecture",
        "abstract",
        "vintage",
        "travel",
        "people",
        "music",
    ]
    
    # Choose a random keyword from the list
    keyword = random.choice(keywords)
    
    # Set up the Unsplash API endpoint
    base_url = "https://api.unsplash.com/photos/random"
    params = {
        "query": keyword,
        "client_id": api_key,
        "orientation": "landscape",  # Adjust the orientation as needed
        "content_filter": "high",    # Adjust the content filter as needed,
        "count": 30
    }
    
    try:
        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        data = response.json()
        
        # Extract the image URL and description
        image_url = data.get("urls", {}).get("regular", "")
        description = data.get("description", f"A random {keyword} image")
        
        return image_url, description
    except Exception as e:
        print(f"Error fetching image: {e}")
        return None, None

# Generate 100 random image URLs and text descriptions
image_urls = []
text_descriptions = []

for _ in range(100):
    image_url, description = fetch_random_image()
    
    # Check if image_url is None (indicating an error) and skip this iteration
    if image_url is None:
        continue
    
    image_urls.append(image_url)
    text_descriptions.append(description)
    
    # Sleep for 5 seconds before making the next API request
    time.sleep(5)

# Print the generated image URLs and text descriptions
for i, (image_url, description) in enumerate(zip(image_urls, text_descriptions)):
    print(f"Image URL {i + 1}: {image_url}")
    print(f"Text Description {i + 1}: {description}\n")
