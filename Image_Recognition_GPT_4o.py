import json
from dotenv import load_dotenv
import os
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO

# Load from .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def image_to_base64(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to bytes
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        # Encode the bytes to base64 string
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str
def get_chat_response(client, user_text, base64_str):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a famous art historian. Please help me to understand the Sculpture."},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_str}"}
                }
            ]}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

client = OpenAI(api_key=api_key)
base64_str = image_to_base64('./example.jpg')
print(get_chat_response(client, "Could you describe the sculpture from this image?", base64_str))
print(get_chat_response(client, "Could you tell me if the figure in this picture is happy, sad, serious or in another mood?", base64_str))