from math import pi, radians, sqrt, atan2, sin, cos

import numpy as np
from pydantic import BaseModel

from cw.astrodynamics import eccentric_anomaly_from_mean_anomaly, true_anomaly_from_eccentric_anomaly, kepler_to_cartesian
from cw.constants import mu_earth, g_earth

from traj2.environments.launcher_v1 import LauncherV1, Stage, AP_NONE, AP_FLIGHT_PATH_CONTROL, AP_PITCH_CONTROL, AP_PITCH_RATE_CONTROL


class InitialCondition(BaseModel):
    longitude: float
    altitude: float
    theta_e: float
    vie: tuple[float, float]
    target_a: float    
    prop_mass: float
    mu: float
    surface_diameter: float

    @property
    def target_v(self):
        _, target_vii = kepler_to_cartesian( 
            a=self.target_a, e=0.0, i=0., raan=0., omega=0., true_anomaly=0., mu=self.mu)
        return np.linalg.norm(target_vii)
    
    @property
    def target_h(self):
        return self.target_a - self.surface_diameter


class BasicInit(BaseModel):
    longitude: float
    altitude: float
    theta_e: float
    vie: tuple[float, float]
    target_a: float    


class KeplerInit(BaseModel):
    a: float
    e: float
    mean_anomaly: float
    theta_e: float


class TimeToApoapsisInit(BaseModel):
    a: tuple[float, float]
    e: tuple[float, float]
    t: tuple[float, float]
    theta_e: tuple[float, float]



class EnvConfig(BaseModel):
    dt: float = 0.05
    surface_diameter: float = 1737.4e3  #: Diameter means radius, I'm to scared to rename it.
    mu: float = 4.9048695e12
    stages: list[Stage] = [
        Stage(
            dry_mass=2150,
            propellant_mass=2353,
            specific_impulse=311,
            thrust=16_000),
    ]
    gamma_controller_gains: tuple[float, float, float] = (4, 0, 0.2)
    theta_controller_gains: tuple[float, float, float] = (10, 0, 0.0)
    controller_theta_dot_limits: tuple[float, float] = (-1, 1)
    end_at_apogee: bool = False
    end_at_ground: bool = True
    end_at_burnout: bool = False
    init: BasicInit | KeplerInit | TimeToApoapsisInit = BasicInit(
        longitude=0.5 * pi,
        altitude=1,
        theta_e=0.5 * pi,
        vie=(0., 0.),
        target_a=1837.4e3,
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


    def get_init_conditions(self, random: np.random.Generator) -> InitialCondition:
        match self.init:
            case BasicInit():
                return InitialCondition(
                    longitude=self.init.longitude,
                    altitude=self.init.altitude,
                    theta_e=self.init.theta_e,
                    vie=self.init.vie,
                    target_a=self.init.target_a,                    
                    prop_mass=self.stages[0].propellant_mass,
                    mu=self.mu,
                    surface_diameter=self.surface_diameter,
                )
            
            case KeplerInit():
                initial_longitude, initial_altitude, initial_vie = kepler_to_cartesian_2d(
                    a=self.init.a,
                    e=self.init.e,
                    surface_diameter=self.surface_diameter,
                    mean_anomaly=self.init.mean_anomaly,
                    mu=self.mu,
                )

                return InitialCondition(
                    longitude=initial_longitude,
                    altitude=initial_altitude,
                    theta_e=self.init.theta_e,
                    vie=initial_vie,
                    target_a=self.init.a,
                    prop_mass=self.stages[0].propellant_mass,
                    mu=self.mu,
                    surface_diameter=self.surface_diameter,
                )
            
            case TimeToApoapsisInit():
                a = random.uniform(*self.init.a)
                e = random.uniform(*self.init.e)
                t = random.uniform(*self.init.t)
                theta_e = random.uniform(*self.init.theta_e)

                mean_anomaly = pi - t * sqrt(self.mu / (a*a*a))

                initial_longitude, initial_altitude, initial_vie = kepler_to_cartesian_2d(
                    a=a,
                    e=e,
                    surface_diameter=self.surface_diameter,
                    mean_anomaly=mean_anomaly,
                    mu=self.mu,
                )

                max_burn_time = random.normal(t, (self.init.t[1] - self.init.t[0]))
                max_burn_time = 0 if max_burn_time < 0 else max_burn_time

                stage = self.stages[0]
                prop_mass = max_burn_time * stage.thrust / (g_earth * stage.specific_impulse)
                prop_mass = stage.propellant_mass if prop_mass > stage.propellant_mass else prop_mass

                return InitialCondition(
                    longitude=initial_longitude,
                    altitude=initial_altitude,
                    theta_e=theta_e,
                    vie=initial_vie,
                    target_a=a,
                    prop_mass=prop_mass,
                    mu=self.mu,
                    surface_diameter=self.surface_diameter
                )
            
            case _:
                raise Exception("Invalid initial condition configuration.")


def kepler_to_cartesian_2d(a: float, e: float, surface_diameter: float, mean_anomaly: float, mu: float):
    xii, vii = kepler_to_cartesian(
        a=a,
        e=e,
        i=0.,
        raan=0.,
        omega=0.,
        mean_anomaly=mean_anomaly,
        mu=mu
    )

    initial_longitude = atan2(xii[1], xii[0])
    initial_altitude = sqrt(xii[0]*xii[0]+xii[1]*xii[1]) - surface_diameter
    tei = np.array(((-sin(initial_longitude), cos(initial_longitude)),
                    (cos(initial_longitude), sin(initial_longitude))), dtype=np.float64)

    vii = (vii[0], vii[1])
    initial_vie = tei @ vii

    return initial_longitude, initial_altitude, tuple(initial_vie)
