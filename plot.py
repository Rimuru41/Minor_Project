import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['0', '1', '2-21', '22-41', '42-61', '62-81', '82-101']
values = [9077, 6203, 4237, 100, 10, 3, 1]  # Adjusted based on the image

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(categories, values, color='white', edgecolor='black', alpha=1)

# Add labels above bars
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value}', 
            ha='center', va='bottom', fontsize=12, fontname='Times New Roman')

# Labels and title
ax.set_xlabel("Annotation Count", fontsize=14, fontname='Times New Roman')
ax.set_ylabel("Frequency", fontsize=14, fontname='Times New Roman')
ax.set_title("", fontsize=16, fontname='Times New Roman')
ax.set_yticks(np.arange(0, 10000, 1000))
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Show plot
plt.show()
