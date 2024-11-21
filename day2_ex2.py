# Importing necessary libraries
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple




#TODO 
# expolation prob 
# documentation
# clean code check
# test cases 



#physical parameters   
# M => Mass 
# R => Radius

M_Sun:float  = 1.989e30
M_Earth:float = 5.972e24
M_Jupiter:float = 1.898e27
M_Saturn:float = 5.683e26

R_Jupyter:float = 69911e3
R_Saturn:float = 58232e3
R_Earth:float = 6371e3
R_Sun:float = 696340e3

JUPITER_SUN_DISTANCE:float = 778.57e9
SATURN_SUN_DISTANCE:float = 1.429e12
EARTH_SUN_DISTANCE:float = 1.496e11

EARTH_AVERAGE_SPEED:float = 29780.0
JUPITER_AVERAGE_SPEED:float = 13070.0
SATURN_AVERAGE_SPEED:float = 9690.0

GRAVITATIONAL_CONSTANT:float = 6.67430e-11

# Define Planet class
class Planet: 
    def __init__(self, name: str, mass: float, position: np.ndarray, velocity: np.ndarray, radius: float, marker: str):
        self.name = name
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.temp = 0
        self.radius = radius
        self.marker = marker

# Define System class
class System:
    time_step: int = 60 # 1 minute

    # Define a custom colormap
    custom_colors: List[str] = ["red", "purple", "blue"]  # Colors to transition through
    custom_cmap = LinearSegmentedColormap.from_list("RedPurpleBlue", custom_colors)
    
    def __init__(self, time: int = 0):
        self.planets: List[Planet] = []
        self.M_Sun: float = 1.989e30
        self.Pos_Sun: np.ndarray = np.array([0,0])
        self.time = time
        
    def add_planet(self, planet: Planet) -> None:
        self.planets.append(planet)

    def callculate_force_with_sun(self, planet: Planet) -> np.ndarray:

        # calclulate the distance of planet from the sun of the system  
        r = np.linalg.norm(planet.position - self.Pos_Sun)

        # use the formula of F_hat = -G.M.m.r_hat/(r^2)
        return -GRAVITATIONAL_CONSTANT * self.M_Sun * planet.mass / r**3 * (planet.position - self.Pos_Sun) 

    def update_planet_temp(self, planet: Planet) -> None:

        # claclulate the distance of the planet from the sun of the solar system 
        planet.temp =  np.linalg.norm(planet.position - self.Pos_Sun)

    def update_planet(self, planet: Planet) -> None:

        # update the position of the planet with the planet's speed 
        planet.position += planet.velocity * self.time_step
        # update the velocity of the planet with the planet's acceleration
        planet.velocity += self.callculate_force_with_sun(planet) / planet.mass * self.time_step
        self.update_planet_temp(planet)

    def update_system(self) -> None:

        # update the time 
        self.time = self.time + self.time_step

        # update each planet in the system 
        for planet in self.planets:
            self.update_planet(planet)
            
    
    def run(self, n: int) -> np.ndarray:

        # data array to store poses, velocities, and temps 
        trajectory = np.zeros((n, len(self.planets), 5))
        for i in tqdm(range(n)):
            
            #update the systtem 
            self.update_system()

            # gather the new data and store in the trjectory data array 
            for j, planet in enumerate(self.planets):
                trajectory[i, j, :2] = planet.position
                trajectory[i, j, 2:4] = planet.velocity
                trajectory[i, j, 4] = planet.temp

        return trajectory

    def get_random_direction(self) -> np.ndarray:

        # create a normal random vector
        direction = np.random.normal(-1, 1, 2)

        # normalize the vector into a unit vector 
        direction = direction / np.linalg.norm(direction)
        return direction

    def visualize(self, trajectory: np.ndarray) -> None:
        # normalize the temperatures into [0,1] form closes to the farthest 
        colors = (trajectory[:, :, 2] - np.min(trajectory[:, :, 2], axis=(0) )) / (np.max(trajectory[:, :, 2], axis=(0)) - np.min(trajectory[:, :, 2], axis=(0)))
  
        # add title to the plot 
        plt.title("Solar System")

        # config the size of the plot 

        plt.figure(figsize=(10, 10))

        # plotting each planet in the solar system
        for i in range(len(self.planets)):
            # use the log of radius as the size of plnet points
            size = np.log(self.planets[i].radius)

            # use the initalized marker to mark planets in the plot
            marker = self.planets[i].marker

            # plot planets with thier position in the trajectory 
            plt.scatter(trajectory[:, i, 0], trajectory[:, i, 1], s=size, c=colors[:, i], cmap=custom_cmap, marker=marker)

        # plotting the Sun 
        plt.scatter(0, 0, s=np.log(R_Sun), c="yellow", marker="*")

        # adding legend to the plot 
        plt.legend([p.name for p in self.planets] + ["Sun"])

        # adding x and y labels 
        plt.xlabel("X(m)")
        plt.ylabel("Y(m)") 

        # sshowing the plot
        plt.show()



# Create an instance of the System class
solar_system = System()

# Create instances of the Planet class
earth = Planet("Earth",M_Earth,np.array([EARTH_SUN_DISTANCE,0]),EARTH_AVERAGE_SPEED * solar_system.get_random_direction(),R_Earth,"s")
jupiter = Planet("Jupiter",M_Jupiter,np.array([EARTH_SUN_DISTANCE,0]),EARTH_AVERAGE_SPEED * solar_system.get_random_direction(),R_Jupyter,"o")
saturn = Planet("Saturn",M_Saturn,np.array([EARTH_SUN_DISTANCE,0]),EARTH_AVERAGE_SPEED * solar_system.get_random_direction(),R_Saturn,"^")

# Add planets to the solar system
solar_system.add_planet(earth)
solar_system.add_planet(jupiter)
solar_system.add_planet(saturn)

# Run the simulation
trajectory = solar_system.run(365*24*60)


# Visualize the trajectory
solar_system.visualize(trajectory)

# Save the plot as an image
plt.savefig("solar_system.png")
