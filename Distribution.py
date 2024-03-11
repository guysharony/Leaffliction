import sys
import os
import matplotlib.pyplot as plt


def count_jpg_files(category_counts: dict, directory_path: str):
    """
    Count the number of JPG files in the given directory and update
    the category_counts dictionary.

    Args:
        category_counts (dict): a dictionary to store
        {key: category names, value: corresponding .jpg file counts}

        directory_path (str): the path of the directory to count in
    """
    directory_name = os.path.basename(directory_path)
    file_count = len([filename for filename in os.listdir(directory_path)])
    category_counts[directory_name] = file_count


def plot_pie_chart(ax, category_counts: dict, title: str):
    """
    Plot a pie chart showing the distribution of file counts across categories.

    Args:
        ax ('matplotlib.axes._axes.Axes'): the axis to plot the pie chart on
        category_counts (dict): a dictionary to store
        {key: category names, value: corresponding .jpg file counts}
        title (str): title of the pie chart
    """
    labels = list(category_counts.keys())
    counts = list(category_counts.values())
    ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=180)
    ax.set_title(f"{title} Class Distribution")


def plot_bar_chart(ax: "plt.axes._axes.Axes", category_counts: dict):
    """
    Plot a bar chart showing the number of images by category.

    Args:
        ax ('matplotlib.axes._axes.Axes'): the axis to plot the bar chart on
        category_counts (dict): a dictionary to store
        {key: category names, value: corresponding .jpg file counts}
    """
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    colors = plt.cm.tab10(range(len(categories)))
    ax.bar(categories, counts, color=colors)
    ax.set_title("Number of images by category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of images")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")


def display_plots(category_counts: dict, directory_name: str):
    """
    Display a pie chart and a bar chart.

    Args:
        category_counts (dict): a dictionary to store
        {key: category names, value: corresponding .jpg file counts}
        directory_name (str): The name of the directory being analyzed.
    """
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_pie_chart(axes[0], category_counts, directory_name)
    plot_bar_chart(axes[1], category_counts)
    plt.tight_layout()
    plt.show()


def main():
    try:
        assert len(sys.argv) == 2, "Only one argument required."

        directory_path = sys.argv[1]

        assert os.path.exists(
            directory_path
        ), f"Directory {directory_path} does not exist."

        directory_name = os.path.basename(directory_path)

        category_counts = {}

        for root, dirs, files in os.walk(directory_path):
            if len(dirs) == 0:
                count_jpg_files(category_counts, root)

        display_plots(category_counts, directory_name)

    except Exception as error:
        print(f"error: {error}")


if __name__ == "__main__":
    main()
