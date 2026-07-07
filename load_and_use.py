import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import json

    from scipy.interpolate import Rbf

    return Rbf, json, np, plt


@app.cell
def _(Rbf, json, np):
    parameters_array_g  = [0.85, 1, 1.2, 1.5,2]
    parameters_array_h = [0.02, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]

    class ParametricFunction_g:
        def __init__(self, matrix):
            # Convert variable-length rows to (t, x, value) points
            points = []
            for row_idx, row_data in enumerate(matrix):
                x_values = np.linspace(-1, 1, len(row_data))
                for x, val in zip(x_values, row_data):
                    points.append([parameters_array_g[row_idx], x, val])

            points = np.array(points)
            print(points)
            self.rbf = Rbf(points[:, 0], points[:, 1], points[:, 2], 
                           function='thin_plate', smooth=0)

        def __call__(self, t, x):
            return self.rbf(t, x)

        def save(self, filename):
            # Save the exact data points
            data = {
                't': self.rbf.xi[0].tolist(),
                'x': self.rbf.xi[1].tolist(),
                'values': self.rbf.di.tolist()
            }
            with open(filename, 'w') as f:
                json.dump(data, f)

        @classmethod
        def load(cls, filename):
            with open(filename) as f:
                data = json.load(f)

            obj = cls.__new__(cls)
            obj.rbf = Rbf(data['t'], data['x'], data['values'],
                          function='thin_plate', smooth=0)
            return obj

    class ParametricFunction_h:
        def __init__(self, matrix):
            # Convert variable-length rows to (t, x, value) points
            points = []
            for row_idx, row_data in enumerate(matrix):
                x_values = np.linspace(-1, 1, len(row_data))
                for x, val in zip(x_values, row_data):
                    points.append([parameters_array_h[row_idx], x, val])

            points = np.array(points)
            print(points)
            self.rbf = Rbf(points[:, 0], points[:, 1], points[:, 2], 
                           function='thin_plate', smooth=0)

        def __call__(self, t, x):
            return self.rbf(t, x)

        def save(self, filename):
            # Save the exact data points
            data = {
                't': self.rbf.xi[0].tolist(),
                'x': self.rbf.xi[1].tolist(),
                'values': self.rbf.di.tolist()
            }
            with open(filename, 'w') as f:
                json.dump(data, f)

        @classmethod
        def load(cls, filename):
            with open(filename) as f:
                data = json.load(f)

            obj = cls.__new__(cls)
            obj.rbf = Rbf(data['t'], data['x'], data['values'],
                          function='thin_plate', smooth=0)
            return obj

    return ParametricFunction_g, ParametricFunction_h


@app.cell
def _(ParametricFunction_g):
    g_out = ParametricFunction_g.load('func_g.json')
    return (g_out,)


@app.cell
def _(ParametricFunction_h):
    h_out = ParametricFunction_h.load('func_h.json')
    return (h_out,)


@app.cell
def _(g_out):
    g_out([0.85, 0.85],[0,0])
    return


@app.cell
def _(t):
    t 
    return


@app.cell
def _(g_out, h_out, np, plt):
    def g_function(x, parameters):
        '''
        parameters[0] should be between 0.85 and 2. (Experimental parameter (Combination of Top Gate + Bottom Gate))
        parameters[1] is the center of the x_axis
        parameters[2] is the half-length of the domain, positive
        parameters[3] is the amplitude
        '''
        t = parameters[0]
        x_0 = parameters[1]
        r = parameters[2]
        amplitude = parameters[3]
        x_normalized = (x - x_0) / r


        # Smooth window function that tapers to 0 at the boundaries
        # Using cosine taper for smoothness
        abs_x = np.abs(x_normalized)
        window = np.where(abs_x >= 1, 
                          0.0,
                          0.5 * (1 + np.cos(np.pi * abs_x)))

        return amplitude*window * g_out([t]*len(x), x_normalized)

    def h_function(x, parameters):
        '''
        parameters[0] should be between 0.02 and 1.6 (Experimental parameter (Drain Voltage))
        parameters[1] is the center of the x_axis
        parameters[2] is the half-length of the domain, positive
        parameters[3] is the amplitude
        '''
        t = parameters[0]
        x_0 = parameters[1]
        r = parameters[2]
        amplitude = parameters[3]
        x_normalized = (x - x_0) / r


        return amplitude*h_out([t]*len(x), x_normalized)

    N_points = 100
    x_space = np.linspace(-1, 1, N_points)
    plt.plot(x_space, g_function(x_space, [0.85, 0, 1, 1]))
    return N_points, g_function, h_function, x_space


@app.cell
def _(g_function, x_space):
    type(g_function(x_space, [0.85, 0, 1, 1]))
    return


app._unparsable_cell(
    r"""
    (V_TG and V_BG) -> -1 to 1
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    f(x(t), y(t), z(t)) = f(t)
    """,
    name="_"
)


@app.cell
def _(g_function, plt, x_space):
    t = 1.5
    x_0 = 0
    r = 1
    amp = 2

    plt.plot(x_space, g_function(x_space, [0.85, x_0, 1, amp]))
    plt.plot(x_space, g_function(x_space, [0.9, x_0, 1, amp]))
    plt.plot(x_space, g_function(x_space, [1.5, x_0, 1, amp]))
    plt.plot(x_space, g_function(x_space, [1.9, x_0, 1, amp]))
    return (t,)


@app.cell
def _(h_function, plt, x_space):
    plt.plot(x_space, h_function(x_space, [0.02, 0, 1, 1]))
    plt.plot(x_space, h_function(x_space, [4.00, 0, 1, 1]), 'k')
    plt.plot(x_space, h_function(x_space, [0.8, 0, 1, 1]))
    plt.plot(x_space, h_function(x_space, [1.6, 0, 1, 1]))
    return


@app.cell
def _(N_points, h_out, plt, x_space):
    plt.plot(x_space, h_out([0.02]*N_points, x_space))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
