import os
from openai import OpenAI
from pydantic import ValidationError
from app.models.reasoning import ReasoningResponse
from app.utils.prompt_loader import load_prompt
#load api key

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

class ReasoningClient:
    def __init__(self, model: str = 'gpt-4o-mini'):
        self.model = model
        self.prompt_template = load_prompt("reasoning_prompt.txt")
    
    def get_reasoning(self, transcript: str) -> ReasoningResponse:
        prompt = self.prompt_template.replace("{{transcript}}",transcript)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a medical reasoning assistant."},
                {"role": "user", "content": prompt}
                    ],
            response_format={"type": "json_object"}
        )

        json_output = response.choices[0].message.content

        try:
            return ReasoningResponse.model_validate_json(json_output)
        except ValidationError as e:
            raise RuntimeError(f"Response validation failed: {e}")
