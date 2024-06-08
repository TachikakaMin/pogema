import numpy as np
from pydantic import BaseModel, validator
from pogema.utils import grid_to_str

class WarehouseConfig(BaseModel):
    wall_width: int = 5
    wall_height: int = 2
    walls_in_row: int = 5
    walls_rows: int = 5
    bottom_gap: int = 5
    horizontal_gap: int = 1
    vertical_gap: int = 3
    wfi_instances: bool = True
    
    @validator('wall_width', 'wall_height', 'walls_in_row', 'walls_rows', 'bottom_gap', 'horizontal_gap', 'vertical_gap')
    def must_be_positive(cls, value, field):
        if value <= 0:
            raise ValueError(f'{field.name} must be a positive integer')
        return value
    
    @validator('walls_in_row', 'walls_rows')
    def must_be_at_least_one(cls, value, field):
        if value < 1:
            raise ValueError(f'{field.name} must be at least 1')
        return value

def generate_warehouse(cfg: WarehouseConfig):
    height = cfg.vertical_gap * (cfg.walls_rows + 1) + cfg.wall_height * cfg.walls_rows
    width = cfg.bottom_gap * 2 + cfg.wall_width * cfg.walls_in_row + cfg.horizontal_gap * (cfg.walls_in_row - 1)
    
    grid = np.zeros((height, width), dtype=int)
    
    for row in range(cfg.walls_rows):
        row_start = cfg.vertical_gap * (row + 1) + cfg.wall_height * row
        for col in range(cfg.walls_in_row):
            col_start = cfg.bottom_gap + col * (cfg.wall_width + cfg.horizontal_gap)
            grid[row_start:row_start + cfg.wall_height, col_start:col_start + cfg.wall_width] = 1
            
    return grid_to_str(grid)

    
def generate_wfi_positions(grid_str, bottom_gap, vertical_gap):
    if vertical_gap == 1:
        raise ValueError("Cannot generate WFI instance with vertical_gap of 1.")
    
    grid = [list(row) for row in grid_str.strip().split('\n')]
    height = len(grid)
    width = len(grid[0])
    
    start_locations = []
    goal_locations = []
    
    for row in range(1, height - 1):
        if row % 3 == 0:
            continue
        for col in range(bottom_gap - 1):
            if grid[row][col] == '.':
                start_locations.append((row, col))
        for col in range(width - bottom_gap + 1, width):
            if grid[row][col] == '.':
                start_locations.append((row, col))
    
    if vertical_gap == 2:
        for row in range(1, height):
            for col in range(width):
                if grid[row][col] == '.' and grid[row - 1][col] == '#':
                    goal_locations.append((row, col))
    else:
        for row in range(height):
            for col in range(width):
                if grid[row][col] == '.':
                    if (row > 0 and grid[row - 1][col] == '#') or (row < height - 1 and grid[row + 1][col] == '#'):
                        goal_locations.append((row, col))
    
    return start_locations, goal_locations

def generate_wfi_warehouse(cfg: WarehouseConfig):
    grid = generate_warehouse(cfg)
    start_locations, goal_locations = generate_wfi_positions(grid, cfg.bottom_gap, cfg.vertical_gap)
    grid_list = [list(row) for row in grid.split('\n')]
    
    for s in start_locations:
        grid_list[s[0]][s[1]] = '$'
    for s in goal_locations:
        if grid_list[s[0]][s[1]] == '$':
            grid_list[s[0]][s[1]] = '&'
        else:
            grid_list[s[0]][s[1]] = '@'
    str_grid = '\n'.join([''.join(row) for row in grid_list])
    
    return str_grid
