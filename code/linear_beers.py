import matplotlib.pyplot as plt
import numpy as np

DPI = 100

number_of_beers = np.arange(0, 11, 1)  # From 0 to 10 beers
bac = 0.02 * number_of_beers  # BAC as a linear function of the number of beers

plt.figure(figsize=(5, 3))
plt.plot(number_of_beers, bac, marker='o')
plt.title('Blood Alcohol Content vs Number of Beers Consumed')
plt.xlabel('Number of Beers')
plt.ylabel('Blood Alcohol Content (%)')
plt.grid(True)

file_path = "notes/_media/linear-func-bac-vs-beers.png"
plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
plt.close()
