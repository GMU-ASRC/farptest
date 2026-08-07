import sys

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

import pygame
import numpy as np

def matrix_to_color_grid(matrix, walls, start, end, color_grid):
    grid = Grid(matrix=matrix)
    start = grid.node(*start)
    end = grid.node(*end)

    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start, end, grid)

    color_grid.fill(0x777777)
    for pn in path:
        color_grid[pn.y][pn.x] = 0x00ffff

    for wc in walls:
        color_grid[wc[1]][wc[0]] = 0x333333

    color_grid[start.y][start.x] = 0x00ff00
    color_grid[end.y][end.x] = 0xff0000

def astar_sample():
    pygame.init()
    pygame.font.init()

    # --- Config ---
    WIDTH, HEIGHT = 800, 600
    FPS = 120

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pygame Basic Setup")
    clock = pygame.time.Clock()

    side = 0.95 * min(WIDTH, HEIGHT)
    rect = pygame.Rect(0.5 * (WIDTH - side), 0.5 * (HEIGHT - side), side, side)
    matrix = np.ones((80, 80))
    rows, cols = matrix.shape
    cell_size = np.array((side, side)) / matrix.shape
    csz = 0.9 * cell_size

    walls: list[tuple[int, int]] = []
    color_grid = np.zeros_like(matrix)

    font = pygame.font.Font("/home/mabayneh/.local/share/fonts/Rubik-Regular.ttf")
    
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

        mouse_pos = pygame.mouse.get_pos()
        btn1, _, _ = pygame.mouse.get_pressed()
        if btn1 and rect.collidepoint(mouse_pos):
            rel_x, rel_y = mouse_pos[0] - rect[0], mouse_pos[1] - rect[1]
            c0, r0 = int(rel_x / cell_size[0]), int(rel_y / cell_size[1])
            walls.append((c0, r0))
            matrix[r0][c0] = 0


        matrix_to_color_grid(matrix, walls, (0, 0), (cols//2-1, rows//2-1), color_grid)

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

        text_surface = font.render(f"{clock.get_fps():.1f} fps", True, "#ffffff")
        screen.blit(text_surface, (10, 10))
        text_surface = font.render(f"rows={rows}\n  cols={cols}", True, "#dddddd")
        screen.blit(text_surface, (10, 40))
        pygame.display.flip()  # present the frame

    pygame.quit()

if __name__ == "__main__":
    astar_sample()