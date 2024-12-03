# Field and grid sizes

FIELD_WIDTH  = 120
FIELD_HEIGHT = 80

GRID_WIDTH   = 13
GRID_HEIGHT  = 9

SIZE_GRID_FACTOR = ( (GRID_WIDTH / FIELD_WIDTH) * (GRID_HEIGHT / FIELD_HEIGHT) ) ** 0.5

CARRY_CODIFICATION = {
    ( 1,  0): 0,
    ( 1,  1): 1,
    ( 0,  1): 2,
    (-1,  1): 3,
    (-1,  0): 4,
    (-1, -1): 5,
    ( 0, -1): 6,
    ( 1, -1): 7
}

CARRY_DECODIFICATION = {v: k for k, v in CARRY_CODIFICATION.items()}