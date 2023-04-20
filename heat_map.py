import numpy as np
import matplotlib.pyplot as plt


x=[]
y=[]

# Genera dei dati casuali per le posizioni del campo
f = open(r'output\player_pos_heatmap.txt', "r")
for line in f:
    xy = line.split()
    y.append(int(float(xy[1])))
    x.append(int(float(xy[0])))

# Crea la heatmap 2D
heatmap, xedges, yedges = np.histogram2d(x, y, bins=100)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=plt.cm.Reds, vmax=10)
plt.colorbar()
plt.title('Heatmap delle posizioni del campo più utilizzate')
plt.xlabel('Posizione X')
plt.ylabel('Posizione Y')
plt.show()