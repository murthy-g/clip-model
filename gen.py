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

# Process each image
for image_url in image_urls:
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image_input = processor(images=image, return_tensors="pt")

    # Generate random text inputs
    random_texts = [
        "random text " + str(i) for i in range(len(image_urls))
    ]
    text_inputs = processor(text=random_texts, return_tensors="pt")

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**image_input, **text_inputs)

    # Calculate image-text similarity scores
    image_text_similarity = (
        100
        * torch.nn.functional.cosine_similarity(
            outputs.logits_per_image, outputs.logits_per_text
        )
    ).tolist()

    similarities.append(image_text_similarity)

# Print similarity scores for each image-text pair
for i, image_url in enumerate(image_urls):
    print(f"Image URL: {image_url}")
    for j, similarity in enumerate(similarities[i]):
        text_input = random_texts[j]
        print(f"Similarity between '{text_input}' and '{image_url}': {similarity:.2f}")
    print()
