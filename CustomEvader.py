from numpy.typing import NDArray
import numpy as np
import pygame
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from swarmsim.sensors.BinaryFOVSensor import BinaryFOVSensor
from swarmsim.agent.control.AbstractController import AbstractController

from heatmap import Heatmap

V, W = 0.3, 0.6

class CustomEvader(AbstractController):
    def __init__(self, agent, parent=None):
        super().__init__(agent, parent)

        self.goal = self.agent.world.population[0]
        self.dbg_center = self.agent.position + np.asarray([0.7, 0.6]) * (self.goal.position - self.agent.position)
        self.dbg_radius = np.linalg.norm(self.goal.position - self.agent.position) * 0.5
        self.dbg_radius += self.goal.radius
        self.dbg_radius *= 1.1
        self.dbg_rect = (
            *(self.dbg_center - self.dbg_radius),
            self.dbg_radius * 2,
            self.dbg_radius * 2,
        )
        self.heatmap = Heatmap(rect=self.dbg_rect, decay_rate=0.8)
        self.defenders = []

    def get_actions(self, agent,):
        world = agent.world
        self.defenders = [a for a in world.population if a.team == "blue"]
        self.heatmap.update(world, self.defenders, world.dt)
        return 0., 0.

    def draw(self, screen, offset):
        # if not self.agent.is_highlighted:
        #     return

        for d in self.defenders:
            d.is_highlighted = True

        pan, zoom = np.asarray(offset[0]), np.asarray(offset[1])
        self.heatmap.draw(screen, zoom, pan, self.agent.world.population[0])
        pygame.draw.circle(
            screen, "#ff00ff", self.dbg_center * zoom + pan, radius=self.dbg_radius * zoom, width=2)
        pygame.draw.rect(screen, "#00ffff", (
            self.dbg_rect[0] * zoom + pan[0],
            self.dbg_rect[1] * zoom + pan[1],
            self.dbg_rect[2] * zoom,
            self.dbg_rect[3] * zoom,
        ), width=2)

