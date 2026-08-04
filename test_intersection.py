import sys

from swarmsim.util.collider.AABB import AABB
import pygame
import numpy as np

from CustomEvader import getSectorVectors, sectorAABB, sectorRectIntersection

def main():
    pygame.init()

    # --- Config ---
    WIDTH, HEIGHT = 800, 600
    FPS = 60

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pygame Basic Setup")
    clock = pygame.time.Clock()

    rect = (300, 250, 150, 150)
    sensor_origin = np.array((450, 300))
    radius = 100
    angle, span = np.deg2rad(0), np.deg2rad(30)

    move_delta = np.array((5, 5))
    turn_delta = np.deg2rad(1)

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

        if keys[pygame.K_w]:
            sensor_origin[1] -= move_delta[1]
        if keys[pygame.K_s]:
            sensor_origin[1] += move_delta[1]
        if keys[pygame.K_a]:
            sensor_origin[0] -= move_delta[0]
        if keys[pygame.K_d]:
            sensor_origin[0] += move_delta[0]

        if keys[pygame.K_LEFT]:
            angle -= turn_delta
        if keys[pygame.K_RIGHT]:
            angle += turn_delta

        # --- Draw ---
        screen.fill((30, 30, 30))  # background


        e_left, e_right = getSectorVectors(angle, span)
        e_left, e_right = np.asarray(e_left[:2]), np.asarray(e_right[:2])

        langle, rangle = angle + span, angle - span

        pygame.draw.rect(screen, "#8FB7D6", rect, width=2)

        sab = sectorAABB(sensor_origin, radius, langle, rangle)
        sr = (sab[0], sab[1], sab[2] - sab[0], sab[3] - sab[1])
        pygame.draw.rect(screen, "#ff0000", sr, width=1)

        bbox = AABB.from_center_wh(sensor_origin, radius*2).to_rect()
        pygame.draw.arc(screen, "#B9A6D6", bbox, -langle, -rangle, width=2)
        pygame.draw.line(screen, "#E3A08A", sensor_origin, sensor_origin + radius * e_left, width=2)
        pygame.draw.line(screen, "#E3A08A", sensor_origin, sensor_origin + radius * e_right, width=2)
        # pygame.draw.circle(screen, "#A8C17B", sensor_origin, 5.0)

        collided = sectorRectIntersection(rect, sensor_origin, radius, angle, span)
        pygame.draw.rect(screen, "#A8C17B" if collided else "#C95A5A", (700, 500, 50, 50))

        pygame.display.flip()  # present the frame

    pygame.quit()

    print(sensor_origin, radius, langle, rangle)
    print(sensor_origin, radius, langle/np.pi, rangle/np.pi)
    print(sectorAABB(sensor_origin, radius, langle, rangle))

if __name__ == "__main__":
    main()