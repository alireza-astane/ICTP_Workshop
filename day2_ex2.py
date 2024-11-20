import numpy as np 
from tqdm import tqdm

# Importing necessary libraries
import matplotlib.pyplot as plt 

# Define physical parameters
M_Sun = 1.989e30
M_Earth = 5.972e24
M_Jupiter = 1.898e27
M_Saturn = 5.683e26
R_Jupyter = 69911e3
R_Saturn = 58232e3
R_Earth = 6371e3
R_Sun = 696340e3
jupyter_sun_distance = 778.57e9
saturn_sun_distance = 1.429e12
earth_sun_distance = 1.496e11
earth_average_speed = 29780
jupiter_average_speed = 13070
saturn_average_speed = 9690
GravitationalConstant  = 6.67430e-11

# Define Planet class
class Planet: 
    def __init__(self,name,mass,position,velocity,radius):
        self.name = name
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.temp = 0
        self.radius = radius

# Define System class
class System:
    time_step = 60 # 1 minute
    
    def __init__(self,time = 0):
        self.planets = []
        self.M_Sun = 1.989e30
        self.Pos_Sun = np.array([0,0])
        self.time = time
        
    def add_planet(self,planet):
        self.planets.append(planet)

    def callculate_force_with_sun(self,planet):
        r = np.linalg.norm(planet.position - self.Pos_Sun)
        return -GravitationalConstant * self.M_Sun * planet.mass / r**3 * (planet.position - self.Pos_Sun) 

    def update_planet_temp(self,planet):
        planet.temp =  1/np.linalg.norm(planet.position - self.Pos_Sun)

    def update_planet(self,planet):
        planet.position += planet.velocity * self.time_step
        planet.velocity += self.callculate_force_with_sun(planet) / planet.mass * self.time_step
        self.update_planet_temp(planet)

    def update(self):
        for planet in self.planets:
            self.update_planet(planet)
            self.time = self.time + self.time_step
    
    def run(self,n):
        trajectory = np.zeros((n,len(self.planets),6))
        for i in tqdm(range(n)):
            self.update()
            for j,planet in enumerate(self.planets):
                trajectory[i,j,:2] = planet.position
                trajectory[i,j,2] = np.log(planet.radius)
                trajectory[i,j,3:] = planet.temp

        return trajectory

    def get_random_direction(self):
        direction = np.random.normal(-1,1,2)
        direction = direction/ np.linalg.norm(direction)
        return direction

    def visualize(self,trajectory):
        plt.title("Solar System")
        plt.scatter(0,0,s=np.log(R_Sun))
        
        colors =  (trajectory[:,:,3:] - np.min(trajectory[:,:,3:],axis=(0,1) )) / (np.max(trajectory[:,:,3:],axis=(0,1)) - np.min(trajectory[:,:,3:],axis=(0,1)))

        for i in range(len(self.planets)):
            plt.scatter(trajectory[:,i,0],trajectory[:,i,1],s=trajectory[:,i,2],c=colors[:,i])

        plt.legend(["Sun"] + [p.name for p in self.planets])

        plt.xlabel("X(m)")
        plt.ylabel("Y(m)") 

        plt.show()

# Create an instance of the System class
solar_system = System()

# Create instances of the Planet class
earth = Planet("Earth",M_Earth,np.array([earth_sun_distance,0]),earth_average_speed * solar_system.get_random_direction(),R_Earth)
jupiter = Planet("Jupiter",M_Jupiter,np.array([earth_sun_distance,0]),earth_average_speed * solar_system.get_random_direction(),R_Jupyter)
saturn = Planet("Saturn",M_Saturn,np.array([earth_sun_distance,0]),earth_average_speed * solar_system.get_random_direction(),R_Saturn)

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




        

        

    


