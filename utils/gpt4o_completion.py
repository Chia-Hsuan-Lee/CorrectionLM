import openai
from config import CONFIG

"""ChatGPT completion"""
def chatgpt_completion(prompt_text):
    openai.api_key = CONFIG['openai_api_key']
    return  openai.ChatCompletion.create(
        model="gpt-4o-2024-05-13",  
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=200,
        temperature=0,
        stop=['--', ';', '#'],
        )['choices'][0]['message']['content']
