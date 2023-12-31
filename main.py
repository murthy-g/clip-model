import torch
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import random

# Load CLIP model and tokenizer
model_name = "openai/clip-vit-base-patch16"
tokenizer = CLIPTokenizer.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

# Sample image URLs and text descriptions
sample_image_urls = [
    "https://www.kasandbox.org/programming-images/avatars/leaf-grey.png",
    "https://www.kasandbox.org/programming-images/avatars/leaf-orange.png",
    "https://www.kasandbox.org/programming-images/avatars/leaf-red.png",
    "https://www.kasandbox.org/programming-images/avatars/leaf-yellow.png",
]

sample_text_descriptions = [
    'A delicate leaf with subtle gray hues, displaying a sense of tranquility and elegance.',
    'An eye-catching leaf with vibrant orange tones, radiating warmth and energy.',
    'A captivating leaf in rich, deep red shades, evoking feelings of passion and intensity.',
    'A cheerful leaf adorned in bright yellow hues, symbolizing joy and positivity.',
]

# Initialize the CLIP processor
processor = CLIPProcessor.from_pretrained(model_name)

# Prepare a list to store similarity scores
similarities = []

# Process each sample image and text
for i, image_url in enumerate(sample_image_urls):
    text_description = sample_text_descriptions[i]

    # Download and process the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image_input = processor(images=image, return_tensors="pt")

    # Process the text
    text_input = processor(text=text_description, return_tensors="pt")

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**image_input, **text_input)

    # Calculate image-text similarity scores
    image_text_similarity = (
        100
        * torch.nn.functional.cosine_similarity(
            outputs.logits_per_image, outputs.logits_per_text
        )
    ).tolist()

    similarities.append(image_text_similarity)

# Print similarity scores for each image-text pair
for i, image_url in enumerate(sample_image_urls):
    text_input = sample_text_descriptions[i]
    print(f"Image URL: {image_url}")
    print(f"Text Description: {text_input}")
    print(f"Similarity: {similarities[i][0]:.2f}")
    print()
