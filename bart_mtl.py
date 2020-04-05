import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.display.width = 0
pd.options.display.max_rows = None
plt.style.use('classic')

def general():
    x = np.linspace(0, 10, 100)
    fig = plt.figure()
    plt.plot(x, np.sin(x), '-')
    plt.plot(x, np.cos(x), '--');
    return x

def matlab_style():
    x = general()
    # MATLAB-style interface
    plt.figure() # create a plot figure
    # create the first of two panels and set current axis
    plt.subplot(2, 1, 1) # (rows, columns, panel number)
    plt.plot(x, np.sin(x))
    # create the second panel and set current axis
    plt.subplot(2, 1, 2)
    plt.plot(x, np.cos(x));

def object_oriented():
    x = general()
    # Object-oriented interface
    # First create a grid of plots
    # ax will be an array of two Axes objects
    fig, ax = plt.subplots(2)
    # Call plot() method on the appropriate object
    ax[0].plot(x, np.sin(x))
    ax[1].plot(x, np.cos(x))
    plt.show()

def simple_line_plots():
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, 10, 1000)
    # Could be also plt. below
    ax.plot(x, np.sin(x), color='red', label='sin(X)')
    ax.plot(x, x - 1, linestyle='dashed', label='just a line')
    ax.plot(x, x + 1, linestyle=':', label='other line')
    ax.plot(x, np.sin(x - 1), color='g', label='other sin(x)')
    ax.plot(x, np.cos(x), label='cos(x)');
    # Adjusting the Plot: Axes Limits
    # plt.xlim(-1, 11)
    # plt.ylim(-1.5, 7);
    # or with plt.axis() by passing a list that specifies [xmin, xmax, ymin,
    # ymax]
    plt.axis([-1, 11, -1.5, 7])
    plt.axis('tight')
    # Labeling Plots
    plt.title("A Sine Curve")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()
    plt.show()

def simple_scatter_plots():
    plt.style.use('seaborn-whitegrid')
    x = np.linspace(0, 10, 30)
    y = np.sin(x)
    plt.plot(x, y, '-<', color='gray',
             markersize=15, linewidth=4,
             markerfacecolor='white',
             markeredgecolor='gray',
             markeredgewidth=2)
    plt.ylim(-1.2, 1.2);
    plt.show()


def markers():
    rng = np.random.RandomState(0)
    for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
        plt.plot(rng.rand(5), rng.rand(5), marker,
                 label="marker='{0}'".format(marker))
    plt.legend(numpoints=1)
    plt.xlim(0, 1.8)
    plt.show()

# Scatter Plots with plt.scatter
def scatter_with_plt():
    rng = np.random.RandomState(0)
    x = rng.randn(100)
    y = rng.randn(100)
    colors = rng.rand(100)
    sizes = 1000 * rng.rand(100)
    plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
                cmap='viridis')
    plt.colorbar();  # show color scale
    plt.show()


def iris_flowers():
    from sklearn.datasets import load_iris
    iris = load_iris()
    features = iris.data.T
    plt.scatter(features[0], features[1], alpha=0.2,
                s=100 * features[3], c=iris.target, cmap='viridis')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1]);
    plt.show()


def basic_errorbars():
    plt.style.use('seaborn-whitegrid')
    x = np.linspace(0, 10, 50)
    dy = 0.8
    y = np.sin(x) + dy * np.random.randn(50)
    # plt.errorbar(x, y, yerr=dy, fmt='.k');
    plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
                 ecolor='lightgray', elinewidth=3, capsize=0);
    plt.show()


def continous_errors():
    from sklearn.gaussian_process import GaussianProcess
    # define the model and draw some data
    model = lambda x: x * np.sin(x)
    xdata = np.array([1, 3, 5, 6, 8])
    ydata = model(xdata)
    # Compute the Gaussian process fit
    gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1E-1,
                         random_start=100)
    gp.fit(xdata[:, np.newaxis], ydata)
    xfit = np.linspace(0, 10, 1000)
    yfit, MSE = gp.predict(xfit[:, np.newaxis], eval_MSE=True)
    dyfit = 2 * np.sqrt(MSE)  # 2*sigma ~ 95% confidence region
    # Visualize the result
    plt.plot(xdata, ydata, 'or')
    plt.plot(xfit, yfit, '-', color='gray')
    plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                     color='gray', alpha=0.2)
    plt.xlim(0, 10);
    plt.show()

# Visualizing a Three-Dimensional Function
def three_d_functions():
    plt.style.use('seaborn-white')
    def f(x, y):
        return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

    x = np.linspace(0, 5, 50)
    y = np.linspace(0, 5, 40)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    # plt.contour(X, Y, Z, 20, cmap='RdGy')
    # plt.contourf(X, Y, Z, 20, cmap='RdGy')
    # plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
    #            cmap='RdGy')
    # plt.axis(aspect='image');
    contours = plt.contour(X, Y, Z, 3, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
               cmap='RdGy', alpha=0.5)
    plt.colorbar();
    plt.show()


def the_hist():
    plt.style.use('seaborn-white')
    # data = np.random.randn(1000)
    # plt.hist(data)
    # plt.hist(data, bins=30, normed=True, alpha=0.5,
    #          histtype='stepfilled', color='steelblue',
    #          edgecolor='none');
    x1 = np.random.normal(0, 0.8, 1000)
    x2 = np.random.normal(-2, 1, 1000)
    x3 = np.random.normal(3, 2, 1000)
    kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)
    plt.hist(x1, **kwargs)
    plt.hist(x2, **kwargs)
    plt.hist(x3, **kwargs);
    plt.show()

def two_d_hist():
    plt.style.use('seaborn-white')
    mean = [0, 0]
    cov = [[1, 1], [1, 2]]
    x, y = np.random.multivariate_normal(mean, cov, 10000).T
    # plt.hist2d(x, y, bins=30, cmap='Blues')
    plt.hexbin(x, y, gridsize=30, cmap='Blues')
    cb = plt.colorbar(label='count in bin')
    cb.set_label('counts in bin')
    plt.show()

def plot_legend():
    x = np.linspace(0, 10, 1000)
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x), '-b', label='Sine')
    ax.plot(x, np.cos(x), '--r', label='Cosine')
    ax.axis('equal')
    ax.legend(loc='upper left', frameon=False)
    plt.show()



def california():
    cities = pd.read_csv('notebooks/data/california_cities.csv')
    print(cities.head())
    # Extract the data we're interested in
    lat, lon = cities['latd'], cities['longd']
    population, area = cities['population_total'], cities['area_total_km2']
    # Scatter the points, using size and color but no label
    plt.scatter(lon, lat, label=None,
                c=population, cmap='viridis',
                s=area, linewidth=0, alpha=0.5)
    plt.axis(aspect='equal')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.colorbar(label='log$_{10}$(population)')
    # plt.clim(3, 7)
    # Here we create a legend:
    # we'll plot empty lists with the desired size and label
    for area in [100, 300, 500]:
        plt.scatter([], [], c='k', alpha=0.3, s=area,
                    label=str(area) + ' km$^2$')
    plt.legend(scatterpoints=1, frameon=False,
               labelspacing=1, title='City Area')
    plt.title('California Cities: Area and Population');
    plt.show()

# Multiple subplots
# plt.axes: Subplots by Hand
def multiple_hand():
    ax1 = plt.axes()  # standard axes
    ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
    plt.show()

# The equivalent of this command within the object-oriented interface is
# fig.add_axes(). Let’s use this to create two vertically stacked axes
def multiple_object():
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[], ylim=(-1.2, 1.2))
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4], ylim=(-1.2, 1.2))
    x = np.linspace(0, 10)
    ax1.plot(np.sin(x))
    ax2.plot(np.cos(x))
    plt.show()

# plt.subplot: Simple Grids of Subplots
# creates a single subplot within a grid. As you can see, this
# command takes three integer arguments—the number of rows, the number of columns,
# and the index of the plot to be created in this scheme, which runs from the
# upper left to the bottom right
def multiple_grids_objects():
    for i in range(1, 7):
        plt.subplot(2, 3, i)
        plt.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
    plt.show()

# The command plt.subplots_adjust can be used to adjust the spacing between
# these plots.
def multiple_grids_objects_2():
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(1, 7):
        ax = fig.add_subplot(2, 3, i)
        ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')
    plt.show()

# We’ve used the hspace and wspace arguments of plt.subplots_adjust, which specify
# the spacing along the height and width of the figure, in units of the subplot size (in
# this case, the space is 40% of the subplot width and height).

# plt.subplots: The Whole Grid in One Go
# Rather than creating a single subplot, this function creates a full grid of
# subplots in a single line, returning them in a NumPy array.
def whole_grid():
    fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
    # axes are in a two-dimensional array, indexed by [row, col]
    for i in range(2):
        for j in range(3):
            ax[i, j].text(0.5, 0.5, str((i, j)), fontsize=18, ha='center')
    plt.show()

# plt.GridSpec: More Complicated Arrangements
# To go beyond a regular grid to subplots that span multiple rows and columns,
# plt.GridSpec() is the best tool. The plt.GridSpec() object does not create a plot by
# itself; it is simply a convenient interface that is recognized by the plt.subplot()
# command. For example, a gridspec for a grid of two rows and three columns with
# some specified width and height space looks like this:
def multiple_complic():
    grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
    # From this we can specify subplot locations and extents using the familiar Python slicing
    # syntax
    plt.subplot(grid[0, 0])
    plt.subplot(grid[0, 1:])
    plt.subplot(grid[1, :2])
    plt.subplot(grid[1, 2])
    plt.show()

def multiple_complic_example():
    # Create some normally distributed data
    mean = [0, 0]
    cov = [[1, 1], [1, 2]]
    x, y = np.random.multivariate_normal(mean, cov, 3000).T
    # Set up the axes with gridspec
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[:-1, 1:])
    y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
    # scatter points on the main axes
    main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)
    # histogram on the attached axes
    x_hist.hist(x, 40, histtype='stepfilled',
                orientation='vertical', color='gray')
    x_hist.invert_yaxis()
    y_hist.hist(y, 40, histtype='stepfilled',
                orientation='horizontal', color='gray')
    y_hist.invert_xaxis()
    plt.show()







