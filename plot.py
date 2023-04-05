import matplotlib.pyplot as plt


def plot_lines(x_data, y_data, x_label, y_label, line_names):
    """
    Plot multiple lines using Matplotlib library.

    Args:
        x_data (list): List of x-axis data.
        y_data (list): List of y-axis data.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        line_names (list): List of line names to be used in legend.

    Returns:
        None
    """

    # Create a new figure and set its size
    fig = plt.figure(figsize=(10, 5))

    # Plot each line with its corresponding name
    for i in range(len(y_data)):
        plt.plot(x_data, y_data[i], label=line_names[i], color="red")

    # Set x and y labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Set legend
    plt.legend()

    # Show the plot
    plt.show()


def plot_lines_with_truncated_yaxis(x_data, y_data, x_label, y_label, line_names, y_min=0, y_max=None):
    """
    Plot multiple lines using Matplotlib library with truncated y-axis.

    Args:
        x_data (list): List of x-axis data.
        y_data (list): List of y-axis data.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        line_names (list): List of line names to be used in legend.
        y_min (float): Minimum value of y-axis. Default value is 0.
        y_max (float): Maximum value of y-axis. If not provided, the maximum value in y_data will be used.

    Returns:
        None
    """

    # Set the maximum y-axis value
    if y_max is None:
        y_max = max([max(y) for y in y_data])

    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Plot the lines
    for i in range(len(y_data)):
        ax.plot(x_data, y_data[i], label=line_names[i])

    # Set the x and y-axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Set the y-axis limits
    ax.set_ylim(y_min, y_max)

    # Set the x-axis ticks
    ax.set_xticks(x_data)

    # Set the spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    x_label = "m = n = k"
    y_label = "time cost (ms)"
    x_data = [128, 256, 512, 1024, 2048]
    y_data = [
        [0.014510, 0.067491, 0.480309, 4.386821, 39.962578],
        [0.011943, 0.045511, 0.29771, 2.576606, 24.606461],
        [0.052332, 0.085298, 0.150201, 0.662864, 4.587993],
        [0.036388, 0.06443, 0.119829, 0.619933, 4.406665],
        [0.032191, 0.055045, 0.100946, 0.599858, 4.267075],
        [0.006733, 0.012522, 0.056911, 0.499411, 3.647433],
    ]
    line_names = ["baseline", "shared memory", "multiply data per thread", "float4", "double buffer", "cublas"]

    plot_lines_with_truncated_yaxis(x_data, y_data, x_label, y_label, line_names, y_min=0, y_max=6)
