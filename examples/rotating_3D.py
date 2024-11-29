import matplotlib.pyplot as plt
import numpy as np

#  ----------------------------------------------------
#  first start with two vectors - we are basically projecting a point

#  vector v in the x-y plane laying on the x axis
v = np.array([[10], [0]])

# vector w in the x-y plane at 45 degrees from v
w = np.array([[5], [5]])

# project w onto v with the dot product
w_dot_v = np.dot(w.T, v) / np.dot(v.T, v) * v

# print the result
print(f"Projection of w ({w}) onto v ({v}): {w_dot_v * v}")

#  plot the vectors
fig, ax = plt.subplots()
ax.quiver(0, 0, *v, color="r", angles="xy", scale_units="xy", scale=1)
ax.quiver(0, 0, *w, color="b", angles="xy", scale_units="xy", scale=1)
ax.quiver(0, 0, *w_dot_v, color="g", angles="xy", scale_units="xy", scale=1)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# plt.show()


def single_vector_projection(v, w):
    """
    Project vector w onto vector v.
    """
    return np.dot(w.T, v) / np.dot(v.T, v) * v


#  ----------------------------------------------------
#  now let's project a line, an array of vectors, still in the x-y plane


#  make a line of 10 points at 45 degrees from the x axis
W = np.array([[i, i] for i in range(1, 10)])

#  project the line_45 onto the line_on_x_axis
W_dot_V = np.array([single_vector_projection(v, w) for w in W])

print(f"Projection of line_45 (\n{W}) onto line_on_x_axis (\n{v}):")
print(W_dot_V)

#  plot the lines
fig, ax = plt.subplots()
for i in range(len(W)):
    #  random rgb pattern
    random_color = f"#{np.random.randint(0, 0xFFFFFF):06x}"
    ax.quiver(
        0, 0, *W[i], color=random_color, angles="xy", scale_units="xy", scale=1
    )
    ax.quiver(
        0,
        0,
        *W_dot_V[i],
        color=random_color,
        angles="xy",
        scale_units="xy",
        scale=1,
    )

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# plt.show()

#  ----------------------------------------------------
#  now in 3D space

v_3D = np.array([[10], [0], [0]])

W_3D = np.array([[i, 0, i] for i in range(1, 10)])

W_dot_V_3D = np.array([single_vector_projection(v_3D, w) for w in W_3D])

print(f"Projection of line_45 (\n{W_3D}) onto line_on_x_axis (\n{v_3D}):")
print(W_dot_V_3D)

#  plot the lines
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for i in range(len(W_3D)):
    #  random rgb pattern
    random_color = f"#{np.random.randint(0, 0xFFFFFF):06x}"
    ax.quiver(0, 0, 0, *W_3D[i], color=random_color)
    ax.quiver(0, 0, 0, *W_dot_V_3D[i], color=random_color)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

#  axis labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# plt.show()

#  ----------------------------------------------------
#  now in 3D space with a plane

#  make a plane where x = 0
y_plane = np.array([[0, i, j] for i in range(-10, 10) for j in range(-10, 10)])

P_3D = np.array([[i, i, j] for i in range(-10, 10) for j in range(-10, 10)])

#  multiply each dot on the P_3D plane by the corresponding vector in y_plane
P_dot_V_3D = np.array(
    [single_vector_projection(y_plane[i], p) for i, p in enumerate(P_3D)]
)

print(f"Projection of plane (\n{P_3D}) onto y_plane (\n{y_plane}):")
print(P_dot_V_3D)

#  plot the lines
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for i in range(len(P_3D)):
    #  random rgb pattern
    random_color = f"#{np.random.randint(0, 0xFFFFFF):06x}"
    #  plot them as dots now as they're too many
    ax.scatter(*P_3D[i], color=random_color)
    ax.scatter(*P_dot_V_3D[i], color=random_color)
    #  show the plane also with arrows
    ax.quiver(0, 0, 0, *y_plane[i], color=random_color, alpha=0.2)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

#  axis labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
