
import requests
import json_numpy as json

import pickle
import numpy as np
import math

def get_rewards(instruction, image_path, actions):
    # Initialize rewards list
    all_rewards = []
    
    # Get action rewards in batches of 2, so the reward model fits in a RTX4090 with 24GB memory size
    # Change the `batch_size` accordingly if you are using a different GPU
    batch_size = 2
    num_batches = math.ceil(len(actions) / batch_size)
    
    for i in range(num_batches):
        # Get the current batch of actions
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(actions))
        action_batch = actions[start_idx:end_idx]
        
        payload = {
            "instruction": instruction,
            "image_path": image_path,
            "action": action_batch
        }
        
        response = requests.post(f"http://127.0.0.1:3100/process", data=json.dumps(payload))
        response_data = json.loads(response.text)
        
        all_rewards.extend(response_data["rewards"])
    
    return all_rewards

with open('actions.pkl', 'rb') as f:
    args = pickle.load(f)

with open('result.pkl', 'rb') as f:
    expected_result = pickle.load(f)


print("Actions:", args)

result = get_rewards(*args)


print("Expected: ", expected_result)

print("Results:  ", result)

assert result == expected_result, "Invalid Reward Results"


