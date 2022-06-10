from math import pi, cos, sin

from matplotlib import cm
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm

from environment import LauncherV1Orbital
from cw.astrodynamics import kepler_to_cartesian, cartesian_to_kepler


class Plotter:
    def __init__(self, env: LauncherV1Orbital, results: xr.Dataset) -> None:
        self.env = env
        self.results = results

    def alt(self):
        fig = plt.figure()
        self.results.env_h.plot.line(x="t")
        return fig

    def orbit_view(self):
        fig = plt.figure()

        # Diameter is radius here.
        radius = self.env.sim.surface_diameter

        lim_size = radius + 200e3

        moon = plt.Circle((0, 0), radius, color='r')
        ax = plt.gca()
        ax.add_patch(moon)
        
        x = self.results.env_xii.values[:, 0].flatten()
        y = self.results.env_xii.values[:, 1].flatten()
        self.plot_thrust_marked(x, y)

        moon = plt.Circle((0, 0), self.env.target_a, color='b', ls=":", fill=None)
        ax = plt.gca()
        ax.add_patch(moon)

        # orbit = self.orbit(30)
        # plt.plot(orbit[:, 0], orbit[:, 1], "g--")

        ax.set_aspect('equal')
        plt.xlim([-lim_size, lim_size])
        plt.ylim([-lim_size, lim_size])

        return fig

    def plot_thrust_marked(self, x, y):
        f_thrust = np.linalg.norm(self.results.env_fii_thrust.values, axis=-1)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(f_thrust.min(), f_thrust.max())
        lc = LineCollection(segments, cmap='bwr', norm=norm)
        # Set the values used for colormapping
        lc.set_array(f_thrust)
        # lc.set_linewidth(2)
        line = plt.gca().add_collection(lc)
        # fig.colorbar(line)

    def orbital_elements(self):
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        self.results.env_eccentricity.plot.line(x="t")
        plt.axhline(self.env.target_e, color='b', ls=":")

        plt.subplot(2, 1, 2)
        self.results.env_semi_major_axis.plot.line(x="t")
        plt.axhline(self.env.target_a, color='b', ls=":")
        return fig


    def orbit(self, n: int):
        a, e, i, raan, omega, true_anomaly, eccentric_anomaly, mean_anomaly = \
            cartesian_to_kepler(
                r=[
                    self.results.env_xii.values[-1, 0],
                    self.results.env_xii.values[-1, 1],
                    0
                ],
                v=[
                    self.results.env_vii.values[-1, 0],
                    self.results.env_vii.values[-1, 1],
                    0
                ],
                mu=self.env.mu
            )
        print(a, e, i, raan, omega)
        coordinates = []

        omega += 0.5*pi

        for anomaly in np.linspace(0, 2*pi, n):
            xii, _ = kepler_to_cartesian(a, e, i, raan, omega, eccentric_anomaly=anomaly, mu=self.env.mu)
            coordinates.append([xii[0] + cos(omega+0.5*pi) * e * a, xii[1] + sin(omega) * e * a])

        return np.array(coordinates)
        

