# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple


# TODO
# documentation
# test cases
# module


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
    def __init__(self, M_Sun: float, time: int = 0, time_step: int = 60):
        """
        constructor method
        """
        self.planets: List[Planet] = []
        self.M_Sun: float = M_Sun
        self.Pos_Sun: np.ndarray = np.array([0, 0])
        self.time: float = time
        self.time_step: float = time_step
        self.kinetic_Energy = 0
        self.potential_Energy = 0

    def add_planet(self, planet: Planet) -> None:
        """
        method to add a specific planet into the systme's list of planets

        Parameters
        ----------
        planet : Planet
            a Planet object to be added into the systems's planets list

        Returns
        -------
        None
        """
        self.planets.append(planet)

    def callculate_Force_and_potential_energy(self, planet: Planet) -> np.ndarray:
        """
        method to claculate the Force and potential energy of a specific planet using Newtons Gravity Law and return them.
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
        method for updating the attrebutes of the planet in the system.
        # update the position of the planet with the planet's speed
        # calculate the accelration and the potential energy
        # update the velocity of the planet with the planet's acceleration
        # claclulate the distance of the planet from the sun of the solar system to represent as the temp
        # compute the kinetic energy of the planet

        Parameters
        ----------


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
        """ """

        # update the time
        self.time = self.time + self.time_step

        # update each planet in the system
        for planet in self.planets:
            self.update_planet(planet)

    def run(self, n: int) -> np.ndarray:
        """ """

        # data array to store poses, velocities, temps , time , Kinetic energy and Potential energy
        trajectory: np.ndarray = np.zeros((n, len(self.planets), 8))
        for i in tqdm(range(n)):
            # update the systtem
            self.update_system()

            # gather the new data and store in the trjectory data array
            for j, planet in enumerate(self.planets):
                trajectory[i, j, :2] = planet.position
                trajectory[i, j, 2:4] = planet.velocity
                trajectory[i, j, 4] = planet.temp
                trajectory[i, j, 6] = planet.kinetic_Energy
                trajectory[i, j, 7] = planet.potential_Energy

            # set timeline of the system with respect to the time step
            trajectory[i, :, 5] = self.time

        return trajectory

    def get_random_direction(self) -> np.ndarray:
        """ """
        # create a normal random vector
        direction: np.ndarray = np.random.normal(-1, 1, 2)

        # normalize the vector into a unit vector
        direction = direction / np.linalg.norm(direction)
        return direction

    def visualize(self, trajectory: np.ndarray, interval: int) -> None:
        """ """
        custom_colors: List[str] = ["red", "purple", "blue"]
        custom_cmap: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
            "RedPurpleBlue", custom_colors
        )
        # normalize the temperatures into [0,1] form closes to the farthest
        colors: np.ndarray = (
            trajectory[:, :, 4] - np.min(trajectory[:, :, 4], axis=(0))
        ) / (
            np.max(trajectory[:, :, 4], axis=(0))
            - np.min(trajectory[:, :, 4], axis=(0))
        )

        # config the size of the plot
        plt.figure(figsize=(10, 10))

        # add title to the plot
        plt.title("Solar System")

        # plotting each planet in the solar system
        for i in range(len(self.planets)):
            # use the log of radius as the size of plnet points
            size: np.ndarray = np.log(self.planets[i].radius)

            # use the initalized marker to mark planets in the plot
            marker: str = self.planets[i].marker

            # plot planets with thier position in the trajectory
            plt.scatter(
                trajectory[::interval, i, 0],
                trajectory[::interval, i, 1],
                s=size,
                c=colors[::interval, i],
                cmap=self.custom_cmap,
                marker=marker,
            )

        # plotting the Sun
        plt.scatter(0, 0, s=np.log(R_Sun), c="yellow", marker="*")

        # adding legend to the plot
        plt.legend([p.name for p in self.planets] + ["Sun"])

        # adding x and y labels
        plt.xlabel("X(m)")
        plt.ylabel("Y(m)")

        # save the plot as an image
        plt.savefig("solar_system.png")

        # sshowing the plot
        plt.show()

        # plot the kinetic energy, potential energy and mechanical energy of the system
        plt.figure(figsize=(10, 10))
        plt.title("Energy of the Solar System")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy (J)")

        # calculate the sum of kinetic energy, potential energy and mechanical energy of the system
        sum_kinetic_energy = np.sum(trajectory[:, :, 6], axis=1)
        sum_potential_energy = np.sum(trajectory[:, :, 7], axis=1)
        sum_mechanical_energy = sum_kinetic_energy + sum_potential_energy

        # plot the sum of kinetic energy, potential energy and mechanical energy of the system
        plt.plot(trajectory[:, 0, 5], sum_kinetic_energy)
        plt.plot(trajectory[:, 0, 5], sum_potential_energy)
        plt.plot(trajectory[:, 0, 5], sum_mechanical_energy)

        # add legend to the plot
        plt.legend(["Kinetic Energy", "Potential Energy", "Mechanical Energy"])

        # save the plot as an image
        plt.savefig("energy.png")

        # show the plot
        plt.show()


def main():
    # physical parameters
    # M => Mass
    # R => Radius

    M_Sun: float = 1.989e30
    M_Earth: float = 5.972e24
    M_Jupiter: float = 1.898e27
    M_Saturn: float = 5.683e26

    R_Jupyter: float = 69911e3
    R_Saturn: float = 58232e3
    R_Earth: float = 6371e3
    R_Sun: float = 696340e3

    JUPITER_SUN_DISTANCE: float = 778.57e9
    SATURN_SUN_DISTANCE: float = 1.429e12
    EARTH_SUN_DISTANCE: float = 1.496e11

    EARTH_AVERAGE_SPEED: float = 29780.0
    JUPITER_AVERAGE_SPEED: float = 13070.0
    SATURN_AVERAGE_SPEED: float = 9690.0

    # Create an instance of the System class
    solar_system: System = System(1.989e30, 0, 60 * 60)

    # Create instances of the Planet class
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

    # Add planets to the solar system
    solar_system.add_planet(earth)
    solar_system.add_planet(jupiter)
    solar_system.add_planet(saturn)

    # Run the simulation for a year
    trajectory: np.ndarray = solar_system.run(30 * 365 * 24)

    # Visualize the trajectory
    solar_system.visualize(trajectory, 24 * 30)

    # Save the plot as an image
    plt.savefig("solar_system.png")


if __name__ == "__main__":
    main()
