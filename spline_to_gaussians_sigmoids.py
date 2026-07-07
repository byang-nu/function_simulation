# %%
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares

from pathlib import Path

# %%
class LoadedFunction:
    '''Wrapper around the RBF data stored in func_g.json and func_h.json'''
    def __init__(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)

        points = np.column_stack([
            np.asarray(data['t'], dtype=float),
            np.asarray(data['x'], dtype=float)
        ])

        values = np.asarray(data['values'], dtype=float)


        self.rbf = RBFInterpolator(points, values, kernel='thin_plate_spline', smoothing=0)
        
    def __call__(self, t, x):
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)

        points = np.column_stack([t, x])

        return self.rbf(points)

# %%
def make_g_h_functions(filename_g_json, filename_h_json):
    g_func = LoadedFunction(filename_g_json)
    h_func = LoadedFunction(filename_h_json)

    def g_function(x, parameters):
        '''
        parameters[0] should be between 0.85 and 2. [Experimental parameter (Combination of Top Gate + Bottom Gate)]
        parameters[1] is the center of the x_axis
        parameters[2] is the half-length of the domain, positive
        parameters[3] is the amplitude

        output: ndarray of the same shape as x
        '''
        t = parameters[0]
        x_0 = parameters[1]
        r = parameters[2]
        r = max(abs(r), 1e-12)  # Ensure r is positive
        amplitude = parameters[3]
        x_normalized = (x - x_0) / r


        # Smooth window function that tapers to 0 at the boundaries
        # Using cosine taper for smoothness
        abs_x = np.abs(x_normalized)
        window = np.where(abs_x >= 1, 
                        0.0,
                        0.5 * (1 + np.cos(np.pi * abs_x)))

        return amplitude * window * g_func(np.full_like(x, t, dtype=float), x_normalized)
    
    def h_function(x, parameters):
        '''
        parameters[0] should be between 0.02 and 1.6 [Experimental parameter (Drain Voltage)]
        parameters[1] is the center of the x_axis
        parameters[2] is the half-length of the domain, positive
        parameters[3] is the amplitude

        output: ndarray of the same shape as x
        '''
        t = parameters[0]
        x_0 = parameters[1]
        r = parameters[2]
        amplitude = parameters[3]
        x_normalized = (x - x_0) / r


        return amplitude * h_func(np.full_like(x, t, dtype=float), x_normalized)
    
    return g_function, h_function

# %%
def load_csv(filename, max_points=None):
    '''To read the activation function .csv files'''
    
    df = pd.read_csv(filename)  # header=0 by default

    # Error handling for missing columns
    if df.shape[1] < 2:
        raise ValueError(f"{filename} must have two columns: input and output")

    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(float)

    # Error handling for missing or non-numeric data
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{filename} contains non-numeric or missing values")
    
    # Sort by x 
    sort_orders = np.argsort(x)
    x = x[sort_orders]
    y = y[sort_orders]

    # (when max_points!= None) use fewer data points to speed up faster
    if max_points != None and len(x) > max_points:
        chosen_indices = np.linspace(0, len(x)-1, max_points).round().astype(int)
        x = x[chosen_indices]
        y = y[chosen_indices]
    return x, y

# %%
def assign_parameters(parameters, num_g, num_h):
    '''To assign the parameters to g and h functions'''

    # Error handling for incorrect number of parameters
    if len(parameters) != 4*(num_g + num_h):
        raise ValueError(f"Expected {4*(num_g + num_h)} parameters, but got {len(parameters)}")
    
    g_params = []
    h_params = []

    index = 0
    for i in range(num_g):
        g_params.append(parameters[index : index+4])
        index += 4

    for i in range(num_h):
        h_params.append(parameters[index : index+4])
        index += 4 

    return g_params, h_params    

# %%
def simulate(x, parameters, num_g, num_h, g_function, h_function):
    '''To simulate the output of the function given the parameters'''

    y_simulated = np.zeros_like(x, dtype=float)
    g_params, h_params = assign_parameters(parameters, num_g, num_h)

    for params in g_params:
        y_simulated += g_function(x, params)

    for params in h_params:
        y_simulated += h_function(x, params)

    return y_simulated

# %%
def calculate_scores(y_true, y_simulated):
    '''To calculate the errors between the true output and the simulated output'''
    rmse = root_mean_squared_error(y_true, y_simulated)
    mse = mean_squared_error(y_true, y_simulated)
    r2 = r2_score(y_true, y_simulated)

    return rmse, mse, r2

# %%
def boundaries(x, y, num_g, num_h):
    '''To set the boundaries for the parameters based on the data'''
    
    xmin, xmax = float(np.min(x)), float(np.max(x))
    width = max(xmax - xmin, 1e-9)
    ymin, ymax = float(np.min(y)), float(np.max(y))
    y_range = max(ymax - ymin, float(np.std(y)), 1.0)

    lower_bounds = []
    upper_bounds = []

    for i in range(num_g):
        lower_bounds += [0.85, xmin - 0.25*width, width/200, -5*y_range]
        upper_bounds += [2.0, xmax + 0.25*width, 1.5*width, 5*y_range]

    for i in range(num_h):
        lower_bounds += [0.02, xmin - 0.25*width, width/200, -5*y_range]
        upper_bounds += [1.6, xmax + 0.25*width, 1.5*width, 5*y_range]
    
    return np.array(lower_bounds), np.array(upper_bounds)

# %%
def random_initial_parameters(lower_bounds, upper_bounds, rng):
    '''To generate random initial parameters within the boundaries'''
    
    return rng.uniform(lower_bounds, upper_bounds)

# %%
def fit_one_model(x, y, num_g, num_h, g_function, h_function, tries, max_nfev=1000, seed=1):
    '''To fit one model and return the results and the errors'''

    lower, upper = boundaries(x, y, num_g, num_h)
    rng = np.random.default_rng(seed)

    # initialize variables to store best parameters and errors
    best_params = None
    best_rmse = np.inf
    best_mse = np.inf
    best_r2 = np.inf

    y_range = max(float(np.max(y) - np.min(y)), float(np.std(y)), 1.0)

    def residuals(params):
        y_simulated = simulate(x, params, num_g, num_h, g_function, h_function)
        return (y_simulated - y) / y_range
    
    for i in range(tries):
        init_guess = random_initial_parameters(lower, upper, rng)

        result = least_squares(residuals, init_guess, bounds=(lower, upper), max_nfev=max_nfev)
        #least_squares(fun, x0) --> finds local minimum of the cost function 

        y_simulated = simulate(x, result.x, num_g, num_h, g_function, h_function)
        rmse, mse, r2 = calculate_scores(y, y_simulated)

        if rmse < best_rmse:
            best_rmse = rmse
            best_mse = mse
            best_r2 = r2
            best_params = result.x

    return {
        "num_g": num_g,
        "num_h": num_h,
        "params": best_params,
        "rmse": best_rmse,
        "mse": best_mse,
        "r2": best_r2
    }

# %%
def find_best_model(x, y, g_function, h_function, max_num_g, max_num_h, tries, max_nfev=1000, seed=1, target_r2=0.99):
    '''To find the simplest model with an r2 value larger than the target value and return it along with all other models tried'''

    results = []

    for total_num in range(1, max_num_g + max_num_h + 1):
        for num_g in range(min(total_num, max_num_g), -1, -1):
            num_h = total_num - num_g

            if num_h > max_num_h:
                continue

            print(f"    trying {num_g} g + {num_h} h...")

            one_result = fit_one_model(x=x, y=y, num_g=num_g, num_h=num_h, g_function=g_function, h_function=h_function, tries=tries, max_nfev=max_nfev, seed=seed)
            results.append(one_result)

            if one_result["r2"] >= target_r2:
                print("     good enough")
                return one_result, results

    best_model = min(results, key=lambda r: r["rmse"])
    return best_model, results

# %%
def save_outputs(filename, out_dir_name, x, y, best_model, all_results, g_function, h_function):
    '''To save the fitted curve, model parameters, and fitting plot'''
    stem = Path(filename).stem
    out_dir_name = Path(out_dir_name)
    out_dir_name.mkdir(parents=True, exist_ok=True)

    y_simulated = simulate(x=x, parameters=best_model["params"], num_g=best_model["num_g"], num_h=best_model["num_h"], g_function=g_function, h_function=h_function)

    # Save fitted curve
    fitted_curve_file = out_dir_name / f"{stem}_fit_curve.csv"
    pd.DataFrame({"x": x, "y": y, "y_simulated": y_simulated, "residual": y-y_simulated}).to_csv(fitted_curve_file, index=False)

    # Save parameters
    g_params, h_params = assign_parameters(best_model["params"], best_model["num_g"], best_model["num_h"])
    json_file = out_dir_name / f"{stem}_best_fit.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "input_csv": str(filename),
                "best_model": {
                    "num_g":    best_model["num_g"],
                    "num_h":    best_model["num_h"], 
                    "rmse": best_model["rmse"],
                    "mse":  best_model["mse"],
                    "r2":   best_model["r2"],
                },
                "g_parameters": [
                    {"t": float(p[0]), "x_0": float(p[1]), "r": float(p[2]), "amplitude": float(p[3])}
                    for p in g_params
                ],
                "h_parameters": [
                    {"t": float(p[0]), "x_0": float(p[1]), "r": float(p[2]), "amplitude": float(p[3])}
                    for p in h_params
                ],
                "all_models_tries": [
                    {
                        "num_g":    r["num_g"],
                        "num_h":    r["num_h"],
                        "rmse": r["rmse"],
                        "mse":  r["mse"],
                        "r2":   r["r2"],
                    }
                    for r in all_results
                ],
            },
            f,
            indent=2,
        )

    # Save fitting plot
    plot_file = out_dir_name / f"{stem}_fit.png"
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, label="data")
    plt.plot(x, y_simulated, label="simulated")

    plt.xlabel("input")
    plt.ylabel("output")
    plt.title(f"{stem}: {best_model["num_g"]} g + {best_model["num_h"]} h, [R2={best_model["r2"]:.5f}, RMSE={best_model["rmse"]:.5f}]")

    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=160)
    plt.close()

    return {
        "activation_function":  str(filename),
        "num_g":    best_model["num_g"],
        "num_h":    best_model["num_h"],
        "rmse": best_model["rmse"],
        "mse":  best_model["mse"],
        "r2":   best_model["r2"],
    }

# %%
def fit_curve_file(filename, g_function, h_function, max_num_g, max_num_h, tries, max_nfev, seed, out_dir_name, max_points, target_r2):
    print(f"Fitting {filename}")

    # Find the best-fitting model on a down-scaled spline .csv file 
    # [otherwise the code took too long to run]
    x_partial, y_partial = load_csv(filename, max_points=max_points)

    best_model, all_results = find_best_model(
        x=x_partial,
        y=y_partial,
        g_function=g_function,
        h_function=h_function,
        max_num_g=max_num_g,
        max_num_h=max_num_h,
        tries=tries,
        max_nfev=max_nfev,
        seed=seed,
        target_r2=target_r2
    )

    # Use the whole file for the actual simulation
    x, y = load_csv(filename)
    y_simulated = simulate(x, best_model["params"], best_model["num_g"], best_model["num_h"], g_function, h_function)
    best_model["rmse"], best_model["mse"], best_model["r2"] = calculate_scores(y, y_simulated)

    summary_row = save_outputs(
        filename=filename,
        out_dir_name=out_dir_name,
        x=x,
        y=y,
        best_model=best_model,
        all_results=all_results,
        g_function=g_function,
        h_function=h_function
    )

    print(f"    best model: {summary_row["num_g"]} g + {summary_row["num_h"]} h")
    print(f"    RMSE = {summary_row["rmse"]:.4f},   R2 = {summary_row["r2"]:.4f}")

    return summary_row


