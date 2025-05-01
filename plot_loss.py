import matplotlib.pyplot as plt

# Data provided by the user
data = """
Step	Training Loss
1	3.297100
2	3.210200
3	3.301000
4	3.052800
5	2.841900
6	2.690600
7	2.539900
8	2.267200
9	2.283100
10	2.046000
11	1.715200
12	1.770500
13	2.108100
14	1.824600
15	1.945000
16	1.997100
17	2.055200
18	1.741700
19	1.899100
20	1.869400
21	1.909200
22	1.908400
23	2.073900
24	1.746700
25	1.937700
26	1.850600
27	1.768000
28	2.171800
29	1.705000
30	2.074300
31	1.766300
32	1.611400
33	1.647800
34	1.799800
35	1.954800
36	1.767300
37	1.717200
38	1.772200
39	1.711700
40	2.012300
41	1.706200
42	1.892400
43	1.919700
44	1.822900
45	1.729800
46	1.947700
47	1.838100
48	1.891300
49	1.976600
50	1.819400
51	1.720000
52	1.959000
53	1.944400
54	2.022500
55	1.809300
56	1.875300
57	1.836600
58	1.677700
59	1.809700
60	1.480500
"""

# Parse the data
lines = data.strip().split('\n')
steps = []
losses = []

# Skip the header line
for line in lines[1:]:
    parts = line.split()
    if len(parts) == 2:
        try:
            steps.append(int(parts[0]))
            losses.append(float(parts[1]))
        except ValueError:
            print(f"Skipping invalid line: {line}")

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, marker='o', linestyle='-')

# Add titles and labels
plt.title('Llama 3.1 8B Fine-tuning Training Loss for Stance Detection')
plt.xlabel('Step')
plt.ylabel('Training Loss')
plt.grid(True)

# Show the plot
# plt.show()

# Optionally, save the plot to a file
plt.savefig('training_loss_plot.svg')
print("Plot saved as training_loss_plot.svg")