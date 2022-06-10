import matplotlib.pyplot as plt
import xarray as xr

from environment import LauncherV1Orbital


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

        plt.plot(self.results.env_xii.values[:, 0], 
                self.results.env_xii.values[:, 1])

        moon = plt.Circle((0, 0), self.env.target_a, color='b', ls=":", fill=None)
        ax = plt.gca()
        ax.add_patch(moon)

        ax.set_aspect('equal')
        plt.xlim([-lim_size, lim_size])
        plt.ylim([-lim_size, lim_size])

        return fig

    def orbital_elements(self):
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        self.results.env_eccentricity.plot.line(x="t")
        plt.axhline(self.env.target_e, color='b', ls=":")

        plt.subplot(2, 1, 2)
        self.results.env_semi_major_axis.plot.line(x="t")
        plt.axhline(self.env.target_a, color='b', ls=":")
        return fig
