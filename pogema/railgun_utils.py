import sys

from pydantic import BaseModel

from typing_extensions import Literal

import torch, random
NOT_FOUND_PATH = -1
def get_distance(distance_map, agent_location, goal_location):
    agent_location = (int(agent_location[0]), int(agent_location[1]))
    goal_location = (int(goal_location[0]), int(goal_location[1]))
    if agent_location not in distance_map:
        return NOT_FOUND_PATH
    return distance_map[agent_location][goal_location[0]][goal_location[1]]

def construct_input_feature(
    map_data,
    agent_locations,
    goal_locations,
    distance_map,
    feature_dim,
    feature_type,
    previous_agent_locations=None,
):
    height, width = map_data.shape
    agent_num = agent_locations.shape[0]
    device = agent_locations.device
    input_features = torch.zeros(
        (feature_dim, height, width), dtype=torch.float32, device=device
    )
    if isinstance(map_data, torch.Tensor):
        input_features[0] = map_data.clone().detach().to(dtype=torch.float32, device=device)
    else:
        input_features[0] = torch.tensor(map_data, dtype=torch.float32, device=device)
    # 使用向量化操作设置agent位置和目标位置
    agent_indices = torch.arange(1, agent_num + 1, dtype=torch.float32, device=device)
    input_features[1, agent_locations[:, 0], agent_locations[:, 1]] = agent_indices
    input_features[2, goal_locations[:, 0], goal_locations[:, 1]] = agent_indices
    if feature_dim == 4:
        # 批量计算距离
        distances = torch.zeros(agent_num, dtype=torch.float32, device=device)
        for i in range(agent_num):
            distances[i] = get_distance(distance_map, agent_locations[i], goal_locations[i])
        input_features[3, agent_locations[:, 0], agent_locations[:, 1]] = distances
    if feature_dim == 5:
        # 批量计算距离
        distances = torch.zeros(agent_num, dtype=torch.float32, device=device)
        for i in range(agent_num):
            distances[i] = get_distance(distance_map, agent_locations[i], goal_locations[i])
        if feature_type == "gradient":
            dx = torch.zeros(agent_num, dtype=torch.float32, device=device)
            dy = torch.zeros(agent_num, dtype=torch.float32, device=device)
            for i in range(agent_num):
                left_position = torch.tensor([agent_locations[i, 0] - 1, agent_locations[i, 1]], device=device)
                left_distances = get_distance(distance_map, left_position, goal_locations[i])
                # Vectorized comparison
                if ((agent_locations == left_position).all(dim=1)).any():
                    left_distances = NOT_FOUND_PATH
                delta_left_distances = left_distances - distances[i]
                
                right_position = torch.tensor([agent_locations[i, 0] + 1, agent_locations[i, 1]], device=device)
                right_distances = get_distance(distance_map, right_position, goal_locations[i])
                if ((agent_locations == right_position).all(dim=1)).any():
                    right_distances = NOT_FOUND_PATH
                delta_right_distances = right_distances - distances[i]

                up_position = torch.tensor([agent_locations[i, 0], agent_locations[i, 1] + 1], device=device)
                up_distances = get_distance(distance_map, up_position, goal_locations[i])
                if ((agent_locations == up_position).all(dim=1)).any():
                    up_distances = NOT_FOUND_PATH
                delta_up_distances = up_distances - distances[i]

                down_position = torch.tensor([agent_locations[i, 0], agent_locations[i, 1] - 1], device=device)
                down_distances = get_distance(distance_map, down_position, goal_locations[i])
                if ((agent_locations == down_position).all(dim=1)).any():
                    down_distances = NOT_FOUND_PATH
                delta_down_distances = down_distances - distances[i]
                
                if delta_left_distances > 0 and delta_right_distances > 0:
                    dx[i] = 0
                elif delta_left_distances >= 0 and delta_right_distances < 0:
                    dx[i] = 1
                elif delta_left_distances < 0 and delta_right_distances >= 0:
                    dx[i] = -1
                elif delta_left_distances < 0 and delta_right_distances < 0:
                    dx[i] = random.choice([-1, 1])
                elif delta_left_distances == 0 and delta_right_distances == 0:
                    dx[i] = random.choice([-1, 0, 1])
                elif delta_left_distances == 0 and delta_right_distances > 0:
                    dx[i] = random.choice([0, -1])
                elif delta_left_distances > 0 and delta_right_distances == 0:
                    dx[i] = random.choice([0, 1])
                else:
                    dx[i] = random.choice([-1, 1])
                if delta_down_distances > 0 and delta_up_distances > 0:
                    dy[i] = 0
                elif delta_down_distances >= 0 and delta_up_distances < 0:
                    dy[i] = 1
                elif delta_down_distances < 0 and delta_up_distances >= 0:
                    dy[i] = -1
                elif delta_down_distances < 0 and delta_up_distances < 0:
                    dy[i] = random.choice([-1, 1])
                elif delta_down_distances == 0 and delta_up_distances == 0:
                    dy[i] = random.choice([-1, 0, 1])
                elif delta_down_distances == 0 and delta_up_distances > 0:
                    dy[i] = random.choice([-1, 0])
                elif delta_down_distances > 0 and delta_up_distances == 0:
                    dy[i] = random.choice([0, 1])
                else:
                    dy[i] = random.choice([-1, 1])
            input_features[3, agent_locations[:, 0], agent_locations[:, 1]] = dx
            input_features[4, agent_locations[:, 0], agent_locations[:, 1]] = dy

    if feature_dim == 6:
        # 批量计算距离
        distances = torch.zeros(agent_num, dtype=torch.float32, device=device)
        for i in range(agent_num):
            distances[i] = get_distance(distance_map, agent_locations[i], goal_locations[i])
        input_features[3, agent_locations[:, 0], agent_locations[:, 1]] = distances
        if feature_type == "gradient":
            dx = torch.zeros(agent_num, dtype=torch.float32, device=device)
            dy = torch.zeros(agent_num, dtype=torch.float32, device=device)
            for i in range(agent_num):
                left_position = torch.tensor([agent_locations[i, 0] - 1, agent_locations[i, 1]], device=device)
                left_distances = get_distance(distance_map, left_position, goal_locations[i])
                # Vectorized comparison
                if ((agent_locations == left_position).all(dim=1)).any():
                    left_distances = NOT_FOUND_PATH
                delta_left_distances = left_distances - distances[i]
                
                right_position = torch.tensor([agent_locations[i, 0] + 1, agent_locations[i, 1]], device=device)
                right_distances = get_distance(distance_map, right_position, goal_locations[i])
                if ((agent_locations == right_position).all(dim=1)).any():
                    right_distances = NOT_FOUND_PATH
                delta_right_distances = right_distances - distances[i]

                up_position = torch.tensor([agent_locations[i, 0], agent_locations[i, 1] + 1], device=device)
                up_distances = get_distance(distance_map, up_position, goal_locations[i])
                if ((agent_locations == up_position).all(dim=1)).any():
                    up_distances = NOT_FOUND_PATH
                delta_up_distances = up_distances - distances[i]

                down_position = torch.tensor([agent_locations[i, 0], agent_locations[i, 1] - 1], device=device)
                down_distances = get_distance(distance_map, down_position, goal_locations[i])
                if ((agent_locations == down_position).all(dim=1)).any():
                    down_distances = NOT_FOUND_PATH
                delta_down_distances = down_distances - distances[i]
                
                
                if delta_left_distances > 0 and delta_right_distances > 0:
                    dx[i] = 0
                elif delta_left_distances >= 0 and delta_right_distances < 0:
                    dx[i] = 1
                elif delta_left_distances < 0 and delta_right_distances >= 0:
                    dx[i] = -1
                elif delta_left_distances < 0 and delta_right_distances < 0:
                    dx[i] = random.choice([-1, 1])
                elif delta_left_distances == 0 and delta_right_distances == 0:
                    dx[i] = random.choice([-1, 0, 1])
                elif delta_left_distances == 0 and delta_right_distances > 0:
                    dx[i] = random.choice([0, -1])
                elif delta_left_distances > 0 and delta_right_distances == 0:
                    dx[i] = random.choice([0, 1])
                else:
                    dx[i] = random.choice([-1, 1])
                if delta_down_distances > 0 and delta_up_distances > 0:
                    dy[i] = 0
                elif delta_down_distances >= 0 and delta_up_distances < 0:
                    dy[i] = 1
                elif delta_down_distances < 0 and delta_up_distances >= 0:
                    dy[i] = -1
                elif delta_down_distances < 0 and delta_up_distances < 0:
                    dy[i] = random.choice([-1, 1])
                elif delta_down_distances == 0 and delta_up_distances == 0:
                    dy[i] = random.choice([-1, 0, 1])
                elif delta_down_distances == 0 and delta_up_distances > 0:
                    dy[i] = random.choice([-1, 0])
                elif delta_down_distances > 0 and delta_up_distances == 0:
                    dy[i] = random.choice([0, 1])
                else:
                    dy[i] = random.choice([-1, 1])
            input_features[4, agent_locations[:, 0], agent_locations[:, 1]] = dx
            input_features[5, agent_locations[:, 0], agent_locations[:, 1]] = dy

        else:
            input_features[4, agent_locations[:, 0], agent_locations[:, 1]] = (
                goal_locations[:, 0] - agent_locations[:, 0]
            ).float()
            input_features[5, agent_locations[:, 0], agent_locations[:, 1]] = (
                goal_locations[:, 1] - agent_locations[:, 1]
            ).float()
    if feature_dim == 7:
        input_features[6, previous_agent_locations[:, 0], previous_agent_locations[:, 1]] = agent_indices
    return input_features