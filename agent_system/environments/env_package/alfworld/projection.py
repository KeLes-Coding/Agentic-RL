# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
import re

def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-30:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            actions[i] = actions[i][-30:]

            # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0
            
        # [New] Validate against admissible actions (functional validity)
        if valids[i] == 1 and action_pools and len(action_pools) > i:
             # Robust cleaning for matching
             # Remove trailing punctuation (especially periods)
             clean_extracted = extracted_action.strip(".,;:!")
             
             # Also strip common prefixes user might hallucinate inside the tag if tag extraction was messy
             # though tag extraction logic usually handles this, sometimes "action: ..." remains if regex was used?
             # Here we used simple substring, so "action: " is unlikely unless user wrote "<action>action: ...</action>"
             if clean_extracted.startswith("action:"):
                 clean_extracted = clean_extracted[7:].strip()
                 
             # Admissible actions might not be lowercased in the pool
             pool = set(a.lower() for a in action_pools[i])
             
             if clean_extracted in pool:
                 # Update the action to the clean version so env receives it cleanly
                 actions[i] = clean_extracted
             else:
                 # Try fuzzy match? Or just fail. 
                 # For safety/strictness, if it's not exact match after cleaning, mark invalid.
                 valids[i] = 0

    return actions, valids
