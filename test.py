import numpy as np
from unittest import TestCase
from Solar import Planet, System

eps = 0.001


def almost_equal(a, b):
    return (abs(a - b) / abs(a)) < eps


def test_calculate_Force_and_potential():
    planet1 = Planet(
        "Earth",
        5.972e24,
        np.array([0.0, 227_940_000_000.0]),
        np.array([0.0, 0.0]),
        6371,
        "o",
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

    force, potential = system.calculate_Force_and_potential(planet1)
    assert almost_equal(force.tolist()[0], 1.73316655e15)
    assert almost_equal(force.tolist()[1], -1.52587609e22)
    assert almost_equal(potential, -3.4780823503686664e33)

    force, potential = system.calculate_Force_and_potential(planet2)
    assert almost_equal(force[0], -1.63267874e21)
    assert almost_equal(force[1], 1.73316655e15)
    assert almost_equal(potential, -3.721531874705354e32)


def test_update_planet():
    planet1 = Planet(
        "Earth",
        5.972e24,
        np.array([0.0, 227_940_000_000.0]),
        np.array([0.0, 0.0]),
        6371,
        "o",
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

    system.update_planet(planet1)
    assert planet1.position.tolist() == [0.0, 227940000000.0]
    assert planet1.velocity.tolist() == [1.7412925856903996e-08, -0.15330302292115697]
    assert planet1.temp == 227940000000.0
    assert planet1.kinetic_Energy == 7.017642507458052e22
    assert planet1.potential_Energy == -3.4780823503686664e33

    system.update_planet(planet2)
    assert planet2.position[0] == 227940000000.0
    assert planet2.position[1] == 1444620.0
    assert planet2.velocity[0] == -0.1533031682391851
    assert planet2.velocity[1] == 24076.999999191146
    assert planet2.temp == 227940000004.57782
    assert planet2.kinetic_Energy == 1.8521476631056448e32
    assert planet2.potential_Energy == -3.721531874655651e32


def test_update_system():
    planet1 = Planet(
        "Earth",
        5.972e24,
        np.array([0.0, 227_940_000_000.0]),
        np.array([0.0, 0.0]),
        6371,
        "o",
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

    system.update_system()

    assert system.time == 60

    assert planet1.position.tolist() == [0.0, 227940000000.0]
    assert planet1.velocity.tolist() == [1.7412925856903996e-08, -0.15330302292115697]
    assert planet1.temp == 227940000000.0
    assert planet1.kinetic_Energy == 7.017642507458052e22
    assert planet1.potential_Energy == -3.4780823503686664e33

    assert planet2.position[0] == 227940000000.0
    assert planet2.position[1] == 1444620.0
    assert planet2.velocity[0] == -0.1533031682391851
    assert planet2.velocity[1] == 24076.999999191146
    assert planet2.temp == 227940000004.57782
    assert planet2.kinetic_Energy == 1.8521476631056448e32
    assert planet2.potential_Energy == -3.721531874655651e32


def test_run():
    planet1 = Planet(
        "Earth",
        5.972e24,
        np.array([0.0, 227_940_000_000.0]),
        np.array([0.0, 0.0]),
        6371,
        "o",
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

    trajectory = system.run(10)

    assert trajectory.shape == (10, 2, 8)
    assert trajectory[0, 0, 0] == 0.0
    assert trajectory[0, 0, 1] == 227940000000.0
    assert trajectory[0, 0, 2] == 1.7412925856903996e-08
    assert trajectory[0, 0, 3] == -0.15330302292115697
    assert trajectory[0, 0, 4] == 227940000000.0
    assert trajectory[0, 0, 5] == 60.0
    assert trajectory[0, 0, 6] == 7.017642507458052e22
    assert trajectory[0, 0, 7] == -3.4780823503686664e33

    assert trajectory[0, 1, 0] == 227940000000.0
    assert trajectory[0, 1, 1] == 1444620.0
    assert trajectory[0, 1, 2] == -0.1533031682391851
    assert trajectory[0, 1, 3] == 24076.999999191146
    assert trajectory[0, 1, 4] == 227940000004.57782
    assert trajectory[0, 1, 5] == 60.0
    assert trajectory[0, 1, 6] == 1.8521476631056448e32
    assert trajectory[0, 1, 7] == -3.721531874655651e32


def test_get_random_direction():
    system = System(1.989e30, 696_340_000, time=0, time_step=60)
    direction = system.get_random_direction()
    assert np.abs(np.linalg.norm(direction) - 1) < eps


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


def main():
    test_get_random_direction()
    test_add_planet()
    test_calculate_Force_and_potential()
    test_update_planet()
    test_update_system()
    test_run()


if __name__ == "__main__":
    main()
