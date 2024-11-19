
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm

G  = 6.67430e-11
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

class Planet: 
    def __init__(self,name,mass,position,velocity,radius):
        self.name = name
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.temp = 0
        self.radius = radius

class System:
    G  = 6.67430e-11
    time_step = 60*60
    
    def __init__(self,time = 0):
        self.planets = []
        self.M_Sun = 1.989e30
        self.Pos_Sun = np.array([0,0])
        self.time = time
        
    def add_planet(self,planet):
        self.planets.append(planet)

    def callculate_force_with_sun(self,planet):
        r = np.linalg.norm(planet.position - self.Pos_Sun)
        return -G * self.M_Sun * planet.mass / r**3 * (planet.position - self.Pos_Sun) 

    def update_planet_temp(self,planet):
        planet.temp =  np.linalg.norm(planet.position - self.Pos_Sun)

    def update_planet(self,planet):
        planet.position += planet.velocity * self.time_step
        planet.velocity += self.callculate_force_with_sun(planet) / planet.mass * self.time_step
        self.update_planet_temp(planet)

    def update(self):
        for planet in self.planets:
            self.update_planet(planet)
            self.time = self.time + self.time_step
    
    def run(self,n):
        trajectory = np.zeros((n,len(self.planets),4))
        for i in tqdm(range(n)):
            self.update()
            for j,planet in enumerate(self.planets):
                trajectory[i,j,:2] = planet.position
                trajectory[i,j,2] = np.log(planet.radius)
                # trajectory[i,j,3] = planet.temp

        return trajectory

    def get_random_direction(self):
        direction = np.random.normal(-1,1,2)
        direction = direction/ np.linalg.norm(direction)
        return direction

    def visualize(self,trajectory):
        plt.scatter(0,0,s=np.log(R_Sun))
        
        for i in range(len(self.planets)):
            plt.scatter(trajectory[:,i,0],trajectory[:,i,1],s=trajectory[:,i,2])
        # for j in tqdm(range(trajectory.shape[0])):=trajectory[:,i,2]
        #     plt.scatter(trajectory[j,:,0],trajectory[j,:,1],s=trajectory[j,:,2],c=(1,0,0))

        plt.legend(["Sun"] + [p.name for p in self.planets])
        plt.show()



solar_system = System()


earth = Planet("Earth",M_Earth,np.array([0,earth_sun_distance]),earth_average_speed * solar_system.get_random_direction(),R_Earth)
jupiter = Planet("Jupiter",M_Jupiter,np.array([0,earth_sun_distance]),earth_average_speed * solar_system.get_random_direction(),R_Jupyter)
saturn = Planet("Saturn",M_Saturn,np.array([0,earth_sun_distance]),earth_average_speed * solar_system.get_random_direction(),R_Saturn)

solar_system.add_planet(earth)
solar_system.add_planet(jupiter)
solar_system.add_planet(saturn)





trajectory = solar_system.run(365*24)
solar_system.visualize(trajectory)


print(R_Saturn/R_Earth, R_Jupyter/R_Earth,R_Earth/R_Earth)
print(np.log(R_Saturn), np.log(R_Jupyter),np.log(R_Earth),np.log(R_Sun))


        

        

    

