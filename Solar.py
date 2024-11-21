"""
This module is used to simulate the solar system with the planets and the Sun.
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple


class Planet:
    """
    Represents a planet in the solar system.

    Parameters
    ----------
    name : str
        The name of the planet.
    mass : float
        The mass of the planet.
    position : numpy.ndarray
        The position of the planet in 2D space.
    velocity: numpy.ndarray
        The velocity of the planet in 2D space.
    radius : float
        The radius of the planet.
    marker : str
        The marker used to represent the planet.
    """

    def __init__(
        self,
        name: str,
        mass: float,
        position: np.ndarray,
        velocity: np.ndarray,
        radius: float,
        marker: str,
    ):
        """
        Construct a Planet object.

        Parameters
        ----------
        name : str
            The name of the planet.
        mass : float
            The mass of the planet.
        position : numpy.ndarray
            The position of the planet in 2D space.
        velocity: numpy.ndarray
            The velocity of the planet in 2D space.
        radius : float
            The radius of the planet.
        marker : str
            The marker used to represent the planet.

        Returns
        -------
        None
        """
        self.name: str = name
        self.mass: float = mass
        self.position: np.ndarray = position
        self.velocity: np.ndarray = velocity
        self.temp: float = 0
        self.radius: float = radius
        self.marker: str = marker


class System:
    """
    Represents a solar system with planets and the Sun.

    Parameters
    ----------
    M_Sun : float
        The mass of the Sun.
    time : int
        The time of the solar system.
    time_step : int
        The time step of the solar system.

    Methods
    -------
    add_planet(planet: Planet) -> None:
        Add a planet to the system object.
    callculate_Force_and_potential_energy(planet: Planet) -> np.ndarray:
        Calculate the Force and potential energy of a specific planet using Newtons Gravity Law and return them.
    update_planet(planet: Planet) -> None:
        Update the attrebutes of the planet in the system.
    update_system() -> None:
        Update the system of the solar system.
    run(n: int) -> np.ndarray:
        Update the system for n time steps, store the data in the trajectory and return the trajectory of the system.
    get_random_direction() -> np.ndarray:
        Generate a random direction vector and normalize it to a unit vector.
    visualize(trajectory: np.ndarray, interval: int) -> None:
        Visualize the solar system with the planets and the sun in the plot.
    """

    def __init__(self, M_Sun: float, R_Sun: float, time: int = 0, time_step: int = 60):
        """
        construct a System object with the mass of the Sun, time and time step.
        """
        self.planets: List[Planet] = []
        self.M_Sun: float = M_Sun
        self.Pos_Sun: np.ndarray = np.array([0, 0])
        self.time: float = time
        self.time_step: float = time_step
        self.kinetic_Energy = 0
        self.potential_Energy = 0
        self.R_Sun = R_Sun

    def add_planet(self, planet: Planet) -> None:
        """
        add a planet to the system object.

        Parameters
        ----------
        planet : Planet
            a planet object in the system object with valid position and mass
        """
        self.planets.append(planet)

    def callculate_Force_and_potential_energy(self, planet: Planet) -> np.ndarray:
        """
        claculate the Force and potential energy of a specific planet using Newtons Gravity Law and return them.
        $$
        F_(1,2)_hat = -G.m1.m2.r_(1,2)_hat/(r_(1,2)^2)
        U_(1,2) = -G.m1.m2/r_(1,2)
        $$

        Using SI Gravitational constant

        Initaializing the total energy into zero
        claculating the distance of planet from sun as distance
        calculating the force and potential energy with respect to the Sun
        add the calculated force and potential energy in to the sum force and potential energy

        calculate the force of the other planets on the planet for each planet in the system's planets list
        check planet to avoid calculating for the same planet.according to Newtons's first law, objects cant exert force on themselves
        calculate the distance of the planet from the other planet
        calculate the force of the other planet on the planet for
        calculate the potential energy of the other planet and the planet
        add the calculated force and potential energy in to the sum force and potential energy

        Parameters
        ----------
        planet : Planet
            a planet object in the system object with valid position and mass

        Returns
        -------
        total_Force : numpy.ndarray
            total gravitational force inserted to the plant

        potential_energy : numpy.ndarray
            total gravitational potential of the planet
        """

        GRAVITATIONAL_CONSTANT: float = 6.67430e-11

        total_Force: np.ndarray = 0
        potential_energy = 0

        distance: np.ndarray = planet.position - self.Pos_Sun
        distance_norm: np.ndarray = np.linalg.norm(distance)

        total_Force += (
            -GRAVITATIONAL_CONSTANT
            * self.M_Sun
            * planet.mass
            * distance
            / (distance_norm**3)
        )
        potential_energy += (
            -GRAVITATIONAL_CONSTANT * self.M_Sun * planet.mass / distance_norm
        )

        for other_planet in self.planets:
            if other_planet != planet:
                distance: np.ndarray = planet.position - other_planet.position
                distance_norm: np.ndarray = np.linalg.norm(distance)

                total_Force += (
                    -GRAVITATIONAL_CONSTANT
                    * other_planet.mass
                    * planet.mass
                    * distance
                    / (distance_norm**3)
                )
                potential_energy += (
                    -GRAVITATIONAL_CONSTANT
                    * other_planet.mass
                    * planet.mass
                    / distance_norm
                )

        return total_Force, potential_energy

    def update_planet(self, planet: Planet) -> None:
        """
        update the attrebutes of the planet in the system.
        update the position of the planet with the planet's speed
        calculate the accelration and the potential energy
        update the velocity of the planet with the planet's acceleration
        claclulate the distance of the planet from the sun of the solar system to represent as the temp
        compute the kinetic energy of the planet

        Parameters
        ----------
        planet : Planet
            a planet object in the system object

        Returns
        -------
        None

        """

        planet.position += planet.velocity * self.time_step

        Force, potential_Energy = self.callculate_Force_and_potential_energy(planet)

        planet.velocity += Force / planet.mass * self.time_step

        planet.temp = np.linalg.norm(planet.position - self.Pos_Sun)

        planet.kinetic_Energy = (
            0.5 * planet.mass * (np.linalg.norm(planet.velocity)) ** 2
        )
        planet.potential_Energy = potential_Energy

    def update_system(self) -> None:
        """
        update the system of the solar system.
        update the time
        update each planet in the system
        """

        self.time = self.time + self.time_step

        for planet in self.planets:
            self.update_planet(planet)

    def run(self, n: int) -> np.ndarray:
        """
        update the system for n time steps, store the data in the trajectory and return the trajectory of the system.

        Parameters
        ----------
        n : int
            number of time steps to update the system

        Returns
        -------
        np.ndarray
            trajectory of the system with the poses, velocities, temps , time , Kinetic energy and Potential energy
        """

        trajectory: np.ndarray = np.zeros((n, len(self.planets), 8))
        for i in tqdm(range(n)):

            self.update_system()
            for j, planet in enumerate(self.planets):
                trajectory[i, j, :2] = planet.position
                trajectory[i, j, 2:4] = planet.velocity
                trajectory[i, j, 4] = planet.temp
                trajectory[i, j, 6] = planet.kinetic_Energy
                trajectory[i, j, 7] = planet.potential_Energy

            trajectory[i, :, 5] = self.time

        return trajectory

    def get_random_direction(self) -> np.ndarray:
        """
        generate a random direction vector and normalize it to a unit vector.

        Returns
        -------
        np.ndarray
            a random unit vector
        """
        direction: np.ndarray = np.random.normal(-1, 1, 2)
        direction = direction / np.linalg.norm(direction)
        return direction

    def visualize(self, trajectory: np.ndarray, interval: int) -> None:
        """
        visualize the solar system with the planets and the sun in the plot.
        normalize the temperatures into [0,1] from closes to the farthest planet to the sun.

        plot the Sun
        plot the planets with the colors based on the temperature of the planets
        plot the Kinetic Energy, Potential Energy and Mechanical Energy of the system
        calculate the sum of kinetic energy, potential energy and mechanical energy of the system
        plot the sum of kinetic energy, potential energy and mechanical energy of the system


        Parameters
        ----------
        trajectory : np.ndarray
            the trajectory of the system with the poses, velocities, temps , time , Kinetic energy and Potential energy
        interval : int
            the interval between each point in the trajectory to be plotted in the plot
        """
        custom_colors: List[str] = ["red", "purple", "blue"]
        custom_cmap: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
            "RedPurpleBlue", custom_colors
        )
        colors: np.ndarray = (
            trajectory[:, :, 4] - np.min(trajectory[:, :, 4], axis=(0))
        ) / (
            np.max(trajectory[:, :, 4], axis=(0))
            - np.min(trajectory[:, :, 4], axis=(0))
        )

        plt.figure(figsize=(10, 10))
        plt.title("Solar System")

        for i in range(len(self.planets)):
            size: np.ndarray = np.log(self.planets[i].radius)
            marker: str = self.planets[i].marker
            plt.scatter(
                trajectory[::interval, i, 0],
                trajectory[::interval, i, 1],
                s=size,
                c=colors[::interval, i],
                cmap=custom_cmap,
                marker=marker,
            )

        plt.scatter(0, 0, s=np.log(self.R_Sun), c="yellow", marker="*")
        plt.legend([p.name for p in self.planets] + ["Sun"])
        plt.xlabel("X(m)")
        plt.ylabel("Y(m)")
        plt.savefig("solar_system.png")
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.title("Energy of the Solar System")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy (J)")

        sum_kinetic_energy = np.sum(trajectory[:, :, 6], axis=1)
        sum_potential_energy = np.sum(trajectory[:, :, 7], axis=1)
        sum_mechanical_energy = sum_kinetic_energy + sum_potential_energy

        plt.plot(trajectory[:, 0, 5], sum_kinetic_energy)
        plt.plot(trajectory[:, 0, 5], sum_potential_energy)
        plt.plot(trajectory[:, 0, 5], sum_mechanical_energy)
        plt.legend(["Kinetic Energy", "Potential Energy", "Mechanical Energy"])
        plt.savefig("energy.png")
        plt.show()
