import numpy as np

def save_array(filename, array, name):
    with open(filename, "w") as f:
        f.write(f"float {name}[] = {{\n")
        flat = array.flatten()
        for i, v in enumerate(flat):
            f.write(f"{v}f, ")
            if i % 10 == 0:
                f.write("\n")
        f.write("\n};")

fc1 = np.load("fc1_w.npy")
fc2 = np.load("fc2_w.npy")

save_array("fc1_w.h", fc1, "fc1_weights")
save_array("fc2_w.h", fc2, "fc2_weights")

print("C Headers Created")
