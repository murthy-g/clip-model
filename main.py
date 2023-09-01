import torch
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
from PIL import Image
import requests
from io import BytesIO

# Load CLIP model and tokenizer
model_name = "openai/clip-vit-base-patch16"
tokenizer = CLIPTokenizer.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

# Text input
text_input = [
    'a leaf in gray color',
    'a leaf in orange color',
    'a leaf in red color',
    'a leaf in yellow color',
    'a happy hopper',
    'a cool hopper',
    'a seed',
    'a seedling',
    'a sapling',
    'a tree',
    'a green color leaf',
    'a red color leaf',
    'a yellow color leaf',
    'a grey color leaf'
]

# Image URLs
image_urls = [
    "https://www.kasandbox.org/programming-images/avatars/leaf-grey.png",
    "https://www.kasandbox.org/programming-images/avatars/leaf-orange.png",
    "https://www.kasandbox.org/programming-images/avatars/leaf-red.png",
    "https://www.kasandbox.org/programming-images/avatars/leaf-yellow.png",
    "https://www.kasandbox.org/programming-images/avatars/cs-hopper-happy.png",
    "https://www.kasandbox.org/programming-images/avatars/cs-hopper-cool.png",
    "https://www.kasandbox.org/programming-images/avatars/leafers-seed.png",
    "https://www.kasandbox.org/programming-images/avatars/leafers-seedling.png",
    "https://www.kasandbox.org/programming-images/avatars/leafers-sapling.png",
    "https://www.kasandbox.org/programming-images/avatars/leafers-tree.png",
]

# Initialize the CLIP processor
processor = CLIPProcessor.from_pretrained(model_name)

# Prepare a list to store similarity scores
similarities = []

# Process each text and image pair
for text in text_input:
    text_inputs = processor(text, return_tensors="pt")
    similarities_for_text = []

    for image_url in image_urls:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image_input = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**image_input, **text_inputs)

        similarity = (
            100
            * torch.nn.functional.cosine_similarity(
                outputs.logits_per_image, outputs.logits_per_text
            )
        ).item()
        
        similarities_for_text.append(similarity)
    
    similarities.append(similarities_for_text)

# Print similarity scores for each text-image pair
for i, text in enumerate(text_input):
    for j, image_url in enumerate(image_urls):
        similarity = similarities[i][j]
        print(f"Similarity between '{text}' and '{image_url}': {similarity:.2f}")
