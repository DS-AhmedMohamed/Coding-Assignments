import sys
import numpy as np
import matplotlib.pyplot as plt

# Function to generate dotplot matrix
def dotplot(seqA, seqB, w, s):
    """
    Generate a dotplot matrix for two sequences.

    Args:
    - seqA: First sequence (string)
    - seqB: Second sequence (string)
    - w: Window size (int)
    - s: Stringency (int)

    Returns:
    - Dotplot matrix (NumPy array)
    """
    # Initialize the dotplot matrix
    dp = np.zeros((len(seqA), len(seqB)), dtype=int)

    # Iterate over positions in seqA
    for i in range(len(seqA)):
        # Define the window for seqA
        start_A = max(0, i - (w - 1) // 2)
        end_A = min(len(seqA), i + (w + 1) // 2)

        # Iterate over positions in seqB
        for j in range(len(seqB)):
            # Define the window for seqB
            start_B = max(0, j - (w - 1) // 2)
            end_B = min(len(seqB), j + (w + 1) // 2)

            # Count the matches within the window
            matches = sum(seqA[x] == seqB[y] for x in range(start_A, end_A) for y in range(start_B, end_B))

            # If number of matches is at least s, set dotplot value to 1
            if matches >= s:
                dp[i, j] = 1

    return dp

# Function to create ASCII dotplot
def dotplot2Ascii(dp, seqA, seqB, heading, filename):
    """
    Create an ASCII dotplot from the dotplot matrix and save it to a file.

    Args:
    - dp: Dotplot matrix (NumPy array)
    - labelA: Label for the y-axis (string)
    - labelB: Label for the x-axis (string)
    - heading: Title for the figure (string)
    - filename: Name of the output file (string)
    """
    with open(filename, "w") as file:
        # Write heading
        file.write(heading + "\n\n" + "  ")
        for i in seqB: file.write(i)
        file.write("\n")
        file.write("-+" + "-" * len(seqB) + "\n")

        # Write dotplot
        for i, row in enumerate(dp):
            file.write(seqA[i] + "|")
            for val in row:
                if val == 1:
                    file.write("*")
                else:
                    file.write(" ")
            file.write("\n")

# Function to create graphical dotplot using individual dots
def dotplot2Graphics_individual(dp, labelA, labelB, heading, filename):
    """
    Create a graphical dotplot from the dotplot matrix using Matplotlib with individual dots.

    Args:
    - dp: Dotplot matrix (NumPy array)
    - labelA: Label for the y-axis (string)
    - labelB: Label for the x-axis (string)
    - heading: Title for the figure (string)
    - filename: Name of the output file (string)
    """
    # Create figure
    fig, ax = plt.subplots()
    
    # Plot individual dots
    for i in range(dp.shape[0]):
        for j in range(dp.shape[1]):
            if dp[i, j] == 1:
                ax.plot(j, i, 'k+', markersize=3)  # Black plus symbol for dot
                
    # Set labels and title
    ax.set_ylabel(labelA)
    ax.set_xlabel(labelB)
    ax.xaxis.set_label_position('top')
    ax.set_title(heading)

    fig.set_size_inches(23, 13, forward=True)

    # Set ticks to reflect sequence positions
    ax.set_xticks(np.arange(len(dp)))
    ax.set_yticks(np.arange(len(dp)))
    ax.set_xticklabels(np.arange(1, len(dp)+1), fontsize=6)
    ax.set_yticklabels(np.arange(1, len(dp)+1), fontsize=6)
    
    # Invert y-axis to have sequence A read from top to bottom
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    # Save the figure
    plt.savefig(filename)
    
    # Display the figure
    plt.show()

# Function to create graphical dotplot using imshow
def dotplot2Graphics_imshow(dp, labelA, labelB, heading, filename):
    """
    Create a graphical dotplot from the dotplot matrix using Matplotlib's imshow function.

    Args:
    - dp: Dotplot matrix (NumPy array)
    - labelA: Label for the y-axis (string)
    - labelB: Label for the x-axis (string)
    - heading: Title for the figure (string)
    - filename: Name of the output file (string)
    """
    # Create figure
    fig, ax = plt.subplots()
    
    # Plot the dotplot using imshow
    ax.imshow(dp, cmap='Greens', interpolation='nearest')
    
    # Set labels and title
    ax.set_ylabel(labelA)
    ax.set_xlabel(labelB)
    ax.xaxis.set_label_position('top')
    ax.set_title(heading)

    fig.set_size_inches(23, 13, forward=True)

    # Set ticks to reflect sequence positions
    if len(dp) < 100:
        ax.set_xticks(np.arange(len(dp)))
        ax.set_yticks(np.arange(len(dp)))
        ax.set_xticklabels(np.arange(1, len(dp)+1), fontsize=6)
        ax.set_yticklabels(np.arange(1, len(dp)+1), fontsize=6)
    
    # Invert y-axis to have sequence A read from top to bottom
    ax.xaxis.tick_top()
    
    # Save the figure
    plt.savefig(filename)
    
    # Display the figure
    plt.show()

def main():
    # Parse command line arguments
    w = int(sys.argv[1])
    s = int(sys.argv[2])
    seqA_file = sys.argv[3]
    seqB_file = sys.argv[4]
    title = sys.argv[5]
    output = sys.argv[6]

    # Read sequences from files
    with open(seqA_file, "r") as file:
        seqA = file.read().strip().split('\n', 1)[1].replace('\n', '')
    with open(seqB_file, "r") as file:
        seqB = file.read().strip().split('\n', 1)[1].replace('\n', '')

    # Generate dotplot matrix
    dp = dotplot(seqA, seqB, w, s)

    # Determine whether to use ASCII or graphical dotplot based on output file extension
    if output.endswith('.txt'):
        dotplot2Ascii(dp, str(seqA_file)[:-6], str(seqB_file)[:-6], title, output)
    elif output.endswith(('.png', '.ps', '.pdf')):
        # Choose one of the graphical dotplot functions based on preference
        #dotplot2Graphics_individual(dp, seqA, seqB, title, output)
        dotplot2Graphics_imshow(dp, str(seqA_file)[:-6], str(seqB_file)[:-6], title, output)
    else:
        print("Unsupported output format. Please use .txt, .png, .ps, or .pdf.")

if __name__ == "__main__":
    main()
