import ollama
import json
from pydantic import BaseModel

class CardOutput(BaseModel):
    """Output structure"""
    
    reading: str
    selected_meanings: list[str]
    example_jp: str
    example_translation: str
    image_prompt: str


def get_answer(word):
    """Sends an LLM request and receives a response in json format"""

    response = ollama.chat(
        model='my-model',

        format=CardOutput.model_json_schema(),
        messages=[
            {
                'role': 'system',
                'content': '''You are a Japanese language expert. Your goal is to generate complete educational data in JSON format. 
If information is missing in the input, you MUST generate it yourself using your knowledge.
NEVER return empty strings or empty lists.'''
            },
            {
                'role': 'user',
                'content':
f"""INPUT DATA TO PROCESS:
{word}

JSON STRUCTURE & CONSTRAINTS:
- "reading": Hiragana only.
- "selected_meanings": 1-2 most accurate English meanings.
- "example_jp": Create a unique, natural Japanese sentence (different from the input context).
- "example_translation": English translation of your new sentence.
- "image_prompt": 4-5 concrete visual Danbooru tags for anime stile. No abstract concepts"""
            }
        ],
        options={
            'stop': ['<|im_start|>', '<|im_end|>', 'User:', 'User'],
            "temperature": 0.1
        }
    )

    content = response['message']['content']
    return json.loads(content)

