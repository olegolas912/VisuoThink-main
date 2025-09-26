import json
import os
from typing import List


############################################################################################################
##### Prompt Generator for Visual Navigation tasks using ReACT agent
############################################################################################################ 


def _format_map(map_grid: List[List[str]]) -> str:
    """Render the emoji grid as a plaintext map that text-only models can follow."""
    return "\n".join("".join(row) for row in map_grid)


def _load_map_text(task_path: str) -> str:
    """Return the initial map in text form for the given task."""
    with open(os.path.join(task_path, "solution.json"), "r", encoding="utf-8") as f:
        configuration = json.load(f)
    return _format_map(configuration["initial_map"])


class VisualNavigationCoT:

    def __init__(self, task_path: str) -> None:
        configuration_img_path = f"{task_path}/map.png"
        map_text = _load_map_text(task_path)
        prompt = f"""
Navigation Task: for a provided map, 🏠 is the home as starting point, 🏢 is the office as the destination. ⬜ means the road, 🚧 means the obstacle. There exists one and only one viable route for each map. Each step you choose a direction and move to the end of the continuous road or the destination.

Map (text view):
{map_text}

If you can process images, you may additionally refer to:
<img src='{configuration_img_path}'>

Starting from 🏠, provide the steps to navigate to 🏢.
Let's think step by step.

Each step should be in json format like:
Step x:
```json
{{"direction": <direction>, "steps": <number of steps>}}
```
"""
        self.prompt = prompt


class VisualNavigationVoT:

    def __init__(self, task_path: str) -> None:
        configuration_img_path = f"{task_path}/map.png"
        map_text = _load_map_text(task_path)
        prompt = f"""
Navigation Task: for a provided map, 🏠 is the home as starting point, 🏢 is the office as the destination. ⬜ means the road, 🚧 means the obstacle. There exists one and only one viable route for each map. Each step you choose a direction and move to the end of the continuous road or the destination.

Map (text view):
{map_text}

If you can process images, you may additionally refer to:
<img src='{configuration_img_path}'>

Starting from 🏠, provide the steps to navigate to 🏢.
Visualize the state after each reasoning step.

Each step should be in json format like:
Step x:
```json
{{"direction": <direction>, "steps": <number of steps>}}
```
"""
        self.prompt = prompt


class VisualNavigationPrompt:
    def __init__(self) -> None:
        self.continue_prompt = """
Please provide your next THOUGHT and ACTION. Your ACTION should be in json format like:
```json
{"direction": <direction>, "steps": <number of steps>}
```
"""
    
    def initial_prompt(self, task_path: str) -> str:
        initial_prompt = f'''
Navigation Task: for a provided map, 🏠 is the home as starting point, 🏢 is the office as the destination. ⬜ means the road, 🚧 means the obstacle. There exists one and only one viable route for each map. Each step you choose a direction and move to the end of the continuous road or the destination.
'''
        configuration_img_path = f"{task_path}/map.png"
        map_text = _load_map_text(task_path)

        prompt = initial_prompt
        prompt += f"Here is the map (text view):\n{map_text}\n"
        prompt += f"If you can process images, you may additionally refer to:\n<img src='{configuration_img_path}'>\n"
        prompt += self.continue_prompt

        return prompt
    

    def get_exec_feedback(self, exit_status, exit_message, exit_file_pths, map_text=None) -> str:
        # if execution fails
        visual_prompt = ""
        if map_text:
            visual_prompt += f"Current map state (text view):\n{map_text}\n"

        if exit_file_pths:
            assert len(exit_file_pths) == 1, "Only one file path is expected"
            img_path = exit_file_pths[0]
            visual_prompt += "If you can process images, here is the current map:\n"
            visual_prompt += f"<img src='{img_path}'>\n"

        if not exit_status:
           prompt = f"OBSERVATION: Execution error. Output:\n{exit_message}\nPlease fix the error.\n"
        else:
            prompt = f"OBSERVATION: Execution success. The output is as follows:\n{exit_message}\n"
        prompt += (visual_prompt + self.continue_prompt)
        return prompt
