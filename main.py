from Solar import System, Planet
import numpy as np


def main():
    # physical parameters
    # M => Mass
    # R => Radius

    M_Sun = 1.989e30
    M_Earth = 5.972e24
    M_Jupiter = 1.898e27
    M_Saturn = 5.683e26

    R_Jupyter = 69911e3
    R_Saturn = 58232e3
    R_Earth = 6371e3
    R_Sun = 696340e3

    JUPITER_SUN_DISTANCE = 778.57e9
    SATURN_SUN_DISTANCE = 1.429e12
    EARTH_SUN_DISTANCE = 1.496e11

    EARTH_AVERAGE_SPEED = 29780.0
    JUPITER_AVERAGE_SPEED = 13070.0
    SATURN_AVERAGE_SPEED = 9690.0

    solar_system = System(M_Sun, R_Sun, 0, 60 * 60)

    earth: Planet = Planet(
        "Earth",
        M_Earth,
        EARTH_SUN_DISTANCE * solar_system.get_random_direction(),
        EARTH_AVERAGE_SPEED * solar_system.get_random_direction(),
        R_Earth,
        "s",
    )

    jupiter: Planet = Planet(
        "Jupiter",
        M_Jupiter,
        JUPITER_SUN_DISTANCE * solar_system.get_random_direction(),
        JUPITER_AVERAGE_SPEED * solar_system.get_random_direction(),
        R_Jupyter,
        "o",
    )

    saturn: Planet = Planet(
        "Saturn",
        M_Saturn,
        SATURN_SUN_DISTANCE * solar_system.get_random_direction(),
        SATURN_AVERAGE_SPEED * solar_system.get_random_direction(),
        R_Saturn,
        "^",
    )

    solar_system.add_planet(earth)
    solar_system.add_planet(jupiter)
    solar_system.add_planet(saturn)

    trajectory = solar_system.run(30 * 365 * 24)

    solar_system.visualize(trajectory, 24 * 30)


if __name__ == "__main__":
    main()
