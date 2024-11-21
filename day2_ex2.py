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
        self.name:str = name
        self.mass:float = mass
        self.position:np.ndarray = position
        self.velocity:np.ndarray = velocity
        self.temp:float = 0
        self.radius:float = radius
        self.marker:str = marker

# Define System class
class System:
    time_step: int = 60 # 1 minute

    # Define a custom colormap
    custom_colors: List[str] = ["red", "purple", "blue"]  # Colors to transition through
    custom_cmap:LinearSegmentedColormap = LinearSegmentedColormap.from_list("RedPurpleBlue", custom_colors)
    
    def __init__(self, time: int = 0):
        self.planets: List[Planet] = []
        self.M_Sun: float = 1.989e30
        self.Pos_Sun: np.ndarray = np.array([0,0])
        self.time:float = time
        
    def add_planet(self, planet: Planet) -> None:
        self.planets.append(planet)

    def callculate_force_with_sun(self, planet: Planet) -> np.ndarray:

        # calclulate the distance of planet from the sun of the system  
        r :np.ndarray = planet.position - self.Pos_Sun
        r_norm : np.ndarray = np.linalg.norm(r)

        # use the formula of F_(1,2)_hat = -G.m1.m2.r_(1,2)_hat/(r_(1,2)^2)

        # intialize the force into zero 
        total_Force:np.ndarray = 0 

        # calculate the force of the sun on the planet
        total_Force += -GRAVITATIONAL_CONSTANT * self.M_Sun * planet.mass *r  / (r_norm**3)

        # calculate the force of the other planets on the planet
        for other_planet in self.planets:

            # to avoid calculating for the same planet / Newtons's first law objects cant exert force on themselves
            if other_planet != planet:

                # calculate the distance of the planet from the other planet
                r:np.ndarray = planet.position - other_planet.position
                r_norm : np.ndarray = np.linalg.norm(r)


                # calculate the force of the other planet on the planet
                total_Force += -GRAVITATIONAL_CONSTANT * other_planet.mass * planet.mass *r  / (r**3)
        
        return total_Force

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
        trajectory:np.ndarray = np.zeros((n, len(self.planets), 5))
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
        direction:np.ndarray = np.random.normal(-1, 1, 2)

        # normalize the vector into a unit vector 
        direction = direction / np.linalg.norm(direction)
        return direction

    def visualize(self, trajectory: np.ndarray,interval:int) -> None:
        # normalize the temperatures into [0,1] form closes to the farthest 
        colors:np.ndarray = (trajectory[:, :, 2] - np.min(trajectory[:, :, 2], axis=(0) )) / (np.max(trajectory[:, :, 2], axis=(0)) - np.min(trajectory[:, :, 2], axis=(0)))
  


        # config the size of the plot 
        plt.figure(figsize=(10, 10))

        # add title to the plot 
        plt.title("Solar System")

        # plotting each planet in the solar system
        for i in range(len(self.planets)):
            # use the log of radius as the size of plnet points
            size:np.ndarray = np.log(self.planets[i].radius)

            # use the initalized marker to mark planets in the plot
            marker:str = self.planets[i].marker

            # plot planets with thier position in the trajectory 
            plt.scatter(trajectory[::interval, i, 0], trajectory[::interval, i, 1], s=size, c=colors[::interval, i], cmap=self.custom_cmap, marker=marker)

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
solar_system:System = System()

# Create instances of the Planet class
earth:Planet = Planet("Earth",M_Earth,np.array([EARTH_SUN_DISTANCE,0]),EARTH_AVERAGE_SPEED * solar_system.get_random_direction(),R_Earth,"s")
jupiter:Planet = Planet("Jupiter",M_Jupiter,np.array([EARTH_SUN_DISTANCE,0]),EARTH_AVERAGE_SPEED * solar_system.get_random_direction(),R_Jupyter,"o")
saturn:Planet = Planet("Saturn",M_Saturn,np.array([EARTH_SUN_DISTANCE,0]),EARTH_AVERAGE_SPEED * solar_system.get_random_direction(),R_Saturn,"^")

# Add planets to the solar system
solar_system.add_planet(earth)
solar_system.add_planet(jupiter)
solar_system.add_planet(saturn)

# Run the simulation for a year
trajectory:np.ndarray = solar_system.run(365*24*60)


# Visualize the trajectory
solar_system.visualize(trajectory,60*24)

# Save the plot as an image
plt.savefig("solar_system.png")
