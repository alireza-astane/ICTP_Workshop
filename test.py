import numpy as np
import unittest
from Solar import Planet, System


def test_calculate_Force_and_potential(self):
    force, potential = system.calculate_Force_and_potential(planet1)
    assertEqual(force.tolist(), [0, 0])
    assertEqual(potential, 0)

    force, potential = system.calculate_Force_and_potential(planet2)
    assertAlmostEqual(force[0], 1.631e20, places=1)
    assertAlmostEqual(force[1], 0, places=1)
    assertAlmostEqual(potential, -1.631e11, places=1)


def test_update_planet(self):
    system.update_planet(planet1)
    assertEqual(planet1.position.tolist(), [0, 0])
    assertEqual(planet1.velocity.tolist(), [0, 0])
    assertEqual(planet1.temp, 0)
    assertEqual(planet1.kinetic_Energy, 0)
    assertEqual(planet1.potential_Energy, 0)

    system.update_planet(planet2)
    assertAlmostEqual(planet2.position[0], 227_940_000_000, places=1)
    assertAlmostEqual(planet2.position[1], 24_077, places=1)
    assertAlmostEqual(planet2.velocity[0], 1.631e20 / (6.39e23 * 24_077) * 60, places=1)
    assertAlmostEqual(planet2.velocity[1], 24_077, places=1)
    assertAlmostEqual(planet2.temp, 227_940_000_000, places=1)
    assertAlmostEqual(planet2.kinetic_Energy, 0.5 * 6.39e23 * (24_077**2), places=1)
    assertAlmostEqual(planet2.potential_Energy, -1.631e11, places=1)


def test_update_system(self):
    system.update_system()
    assertEqual(system.time, 60)
    assertAlmostEqual(planet1.position[0], 0, places=1)
    assertAlmostEqual(planet1.position[1], 0, places=1)
    assertAlmostEqual(planet1.velocity[0], 0, places=1)
    assertAlmostEqual(planet1.velocity[1], 0, places=1)
    assertAlmostEqual(planet1.temp, 0, places=1)
    assertAlmostEqual(planet1.kinetic_Energy, 0, places=1)
    assertAlmostEqual(planet1.potential_Energy, 0, places=1)

    assertAlmostEqual(planet2.position[0], 227_940_000_000, places=1)
    assertAlmostEqual(planet2.position[1], 24_077, places=1)
    assertAlmostEqual(planet2.velocity[0], 1.631e20 / (6.39e23 * 24_077) * 60, places=1)
    assertAlmostEqual(planet2.velocity[1], 24_077, places=1)
    assertAlmostEqual(planet2.temp, 227_940_000_000, places=1)
    assertAlmostEqual(planet2.kinetic_Energy, 0.5 * 6.39e23 * (24_077**2), places=1)
    assertAlmostEqual(planet2.potential_Energy, -1.631e11, places=1)


def test_run(self):
    trajectory = system.run(10)
    assertEqual(trajectory.shape, (10, 2, 8))
    assertAlmostEqual(trajectory[0, 0, 0], 0, places=1)
    assertAlmostEqual(trajectory[0, 0, 1], 0, places=1)
    assertAlmostEqual(trajectory[0, 0, 2], 0, places=1)
    assertAlmostEqual(trajectory[0, 0, 3], 0, places=1)
    assertAlmostEqual(trajectory[0, 0, 4], 0, places=1)
    assertAlmostEqual(trajectory[0, 0, 5], 0, places=1)
    assertAlmostEqual(trajectory[0, 0, 6], 0, places=1)
    assertAlmostEqual(trajectory[0, 0, 7], 0, places=1)

    assertAlmostEqual(trajectory[0, 1, 0], 227_940_000_000, places=1)
    assertAlmostEqual(trajectory[0, 1, 1], 24_077, places=1)
    assertAlmostEqual(trajectory[0, 1, 2], 1.631e20 / (6.39e23 * 24_077) * 60, places=1)
    assertAlmostEqual(trajectory[0, 1, 3], 24_077, places=1)
    assertAlmostEqual(trajectory[0, 1, 4], 227_940_000_000, places=1)
    assertAlmostEqual(trajectory[0, 1, 5], 0, places=1)
    assertAlmostEqual(trajectory[0, 1, 6], 0.5 * 6.39e23 * (24_077**2), places=1)
    assertAlmostEqual(trajectory[0, 1, 7], -1.631e11, places=1)


def test_get_random_direction():
    system = System(1.989e30, 696_340_000, time=0, time_step=60)
    direction = system.get_random_direction()
    assert np.linalg.norm(direction) == 1


def test_add_planet():
    planet1 = Planet(
        "Earth", 5.972e24, np.array([0.0, 0.0]), np.array([0.0, 0.0]), 6371, "o"
    )
    planet2 = Planet(
        "Mars",
        6.39e23,
        np.array([227_940_000_000.0, 0.0]),
        np.array([0.0, 24_077.0]),
        3389,
        "o",
    )
    system = System(1.989e30, 696_340_000, time=0, time_step=60)
    system.add_planet(planet1)
    system.add_planet(planet2)
    assert len(system.planets) == 2
    assert system.planets[0].name == "Earth"
    assert system.planets[1].name, "Mars"


test_get_random_direction()
test_add_planet()
