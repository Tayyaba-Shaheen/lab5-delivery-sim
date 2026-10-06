import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. Create 100 delivery records
# ------------------------------------------------

np.random.seed(42)

delivery_times = np.random.normal(
    loc=32.4,
    scale=8.1,
    size=100
)

# Make all values positive
delivery_times = np.maximum(delivery_times, 5)

# Add some obvious long deliveries
delivery_times[67] = 68
delivery_times[70] = 71
delivery_times[74] = 75

data = {
    "OrderID": range(1, 101),
    "DeliveryTime(min)": delivery_times
}

df = pd.DataFrame(data)

# Save CSV
df.to_csv("delivery_times.csv", index=False)

# Load CSV
df = pd.read_csv("delivery_times.csv")

# ------------------------------------------------
# 2. Mean and standard deviation
# ------------------------------------------------

mean_time = df["DeliveryTime(min)"].mean()
std_time = df["DeliveryTime(min)"].std()

print(
    f"Real delivery times -- Mean: "
    f"{mean_time:.2f} min "
    f"Std: {std_time:.2f} min"
)

# ------------------------------------------------
# 3. Simulate 100 expected delivery times
# ------------------------------------------------

simulated_times = np.random.normal(
    loc=mean_time,
    scale=std_time,
    size=100
)

print(
    f"Simulated 100 values from "
    f"Normal(mean={mean_time:.2f}, std={std_time:.2f})"
)

# ------------------------------------------------
# 4. Missing values
# ------------------------------------------------

print(
    "Missing values:",
    df["DeliveryTime(min)"].isnull().sum()
)

# ------------------------------------------------
# 5. Percentiles
# ------------------------------------------------

q25 = df["DeliveryTime(min)"].quantile(0.25)
q50 = df["DeliveryTime(min)"].quantile(0.50)
q75 = df["DeliveryTime(min)"].quantile(0.75)

print(
    f"Percentiles -- 25th: {q25:.1f} "
    f"50th: {q50:.1f} "
    f"75th: {q75:.1f}"
)

# ------------------------------------------------
# 6. IQR outlier detection
# ------------------------------------------------

IQR = q75 - q25

lower_bound = q25 - 1.5 * IQR
upper_bound = q75 + 1.5 * IQR

outliers = df[
    (df["DeliveryTime(min)"] < lower_bound) |
    (df["DeliveryTime(min)"] > upper_bound)
]

print(
    f"IQR outlier bounds: "
    f"{lower_bound:.1f} - {upper_bound:.1f}"
)

print(
    "Outliers detected:",
    len(outliers),
    "deliveries"
)

print(
    "Order IDs:",
    outliers["OrderID"].tolist()
)

# ------------------------------------------------
# 7. Percentage of outliers
# ------------------------------------------------

percentage = (len(outliers) / len(df)) * 100

print(
    f"Percentage flagged as outliers: "
    f"{percentage:.1f}%"
)

# ------------------------------------------------
# 8. Comparison histogram
# ------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["DeliveryTime(min)"],
    bins=15,
    alpha=0.6,
    label="Real Delivery Times"
)

plt.hist(
    simulated_times,
    bins=15,
    alpha=0.6,
    label="Simulated Delivery Times"
)

plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Frequency")
plt.title("Real vs Simulated Delivery Times")
plt.legend()

plt.savefig("delivery_comparison_histogram.png")
plt.show()

# ------------------------------------------------
# 9. Boxplot
# ------------------------------------------------

plt.figure(figsize=(6, 4))

plt.boxplot(df["DeliveryTime(min)"])

plt.ylabel("Delivery Time (minutes)")
plt.title("Delivery Time Boxplot")

plt.savefig("delivery_boxplot.png")
plt.show()

print(
    "\nSaved delivery_comparison_histogram.png, "
    "delivery_boxplot.png"
)