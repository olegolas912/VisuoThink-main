import json
import os
import sys
from datetime import datetime
from autogen.agentchat.contrib.img_utils import (
    gpt4v_formatter,
)
from autogen.oai.client import OpenAIWrapper
from config import llm_config
from utils_misc import print_error
from time import sleep
from copy import deepcopy  


def _supports_image_messages(call_config: dict) -> bool:
    """Return True if every config entry expects GPT-4V style image payloads."""
    config_list = call_config.get("config_list", [])
    if not config_list:
        return True

    for cfg in config_list:
        api_type = (cfg or {}).get("api_type")
        if api_type and api_type.lower().startswith("ollama"):
            return False
        base_url = (cfg or {}).get("base_url", "")
        if isinstance(base_url, str) and "11434" in base_url:
            return False
    return True


def chat_vlm(prompt: str, history_messages = None, temperature: float = 0., retry_times: int = 10):
    call_config = deepcopy(llm_config)
    for s_config in call_config['config_list']:
        s_config['temperature'] = temperature

    use_image_formatter = _supports_image_messages(call_config)

    interval = 1
    for i in range(retry_times):
        try:
            if history_messages is None:
                history_messages = []
            clean_messages = history_messages + [{"role": "user", "content":  prompt}]
            if use_image_formatter:
                dirty_messages = [{'role': mdict['role'], 'content': gpt4v_formatter(mdict['content'])} for mdict in clean_messages]
            else:
                dirty_messages = clean_messages
            
            client = OpenAIWrapper(**call_config)
            response = client.create(
                messages=dirty_messages,
                timeout=600,
            )
            messages = clean_messages + [{"role": "assistant", "content": response.choices[0].message.content}]
            return response.choices[0].message.content, messages
        except Exception as e:
            if 'limit' in str(e):
                sleep(interval)
                interval = min(interval * 2, 60)
            print_error(e)
            if i >= (retry_times - 1):
                raise e


if __name__ == "__main__":
    # print(llm_config)
    print(chat_vlm('Hello! Introduce yourself and tell me a joke.', temperature=0.8)[0])
