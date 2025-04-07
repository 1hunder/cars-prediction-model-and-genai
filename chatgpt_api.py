from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_explanation_vision(input_data: dict, predicted_price: float, shap_base64: str) -> str:
    text_prompt = f"""
This is a SHAP explanation for a car price prediction.
Based on the SHAP plot and car features below, explain in a clear and friendly way why this price was predicted.

Do not use technical terms, but rather explain the most influential features and what might have increased or decreased the price.
Your answer should be only explanatory, without any additional information or context.
Dont say anything, just start explaining.
Don't use ** or any other formatting, don't use bold text too, just plain text.

Car input:
{input_data}

Predicted price: PLN{predicted_price:,.2f}

Explain the most influential features and what might have increased or decreased the price.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": text_prompt },
                    { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{shap_base64}" } }
                ]
            }
        ],
        max_tokens=600
    )

    return response.choices[0].message.content.strip()
