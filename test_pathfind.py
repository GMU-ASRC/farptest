import sys

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

import pygame
import numpy as np


def astar_sample():
    pygame.init()

    # --- Config ---
    WIDTH, HEIGHT = 800, 600
    FPS = 60

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pygame Basic Setup")
    clock = pygame.time.Clock()

    side = 0.95 * min(WIDTH, HEIGHT)
    rect = (0.5 * (WIDTH - side), 0.5 * (HEIGHT - side), side, side)
    matrix = np.ones((10, 10))
    rows, cols = matrix.shape
    cell_size = np.array((side, side)) / matrix.shape
    csz = 0.9 * cell_size

    walls = np.array([
        (1, 1), (2, 2), (3, 3), (2, 0)
    ])
    for wc in walls:
        matrix[wc[1]][wc[0]] = 0

    grid = Grid(matrix=matrix)
    start = grid.node(0, 0)
    end = grid.node(cols-1, rows-1)

    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, runs = finder.find_path(start, end, grid)

    color_grid = np.ones_like(matrix, dtype=int) * 0x777777

    for pn in path:
        color_grid[pn.y][pn.x] = 0x00ffff

    for wc in walls:
        color_grid[wc[1]][wc[0]] = 0x333333

    color_grid[start.y][start.x] = 0x00ff00
    color_grid[end.y][end.x] = 0xff0000
    
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds since last frame

        # --- Handle events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- User input ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
                running = False

        # --- Draw ---
        screen.fill((30, 30, 30))  # background

        pygame.draw.rect(screen, "#ff0000", rect, width=2)

        for r in range(rows):
            for c in range(cols):
                # color = 0x333333
                # if r == 0 and c == 0:
                #     color = "#00ff00"
                # elif r == 49 and c == 49:
                #     color = "#ff0000"

                pygame.draw.rect(
                    screen, int(color_grid[r][c]),
                    (rect[0] + c*cell_size[0],
                        rect[1] + r*cell_size[1],
                        *csz),
                )

        pygame.display.flip()  # present the frame

    pygame.quit()

if __name__ == "__main__":
    astar_sample()