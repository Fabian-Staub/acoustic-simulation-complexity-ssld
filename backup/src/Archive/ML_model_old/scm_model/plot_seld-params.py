import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("model_search/reverberant/model_search_results.csv")

# Remove rows with duplicate parameter counts
df = df.drop_duplicates(subset=["Num Parameters"], keep="first")

# Plot
plt.figure(figsize=(10, 6))

colors = {
    "MLP": "tab:blue",
    "CNN": "tab:orange",
    "CNN2D": "tab:green"
}

for model_type, group in df.groupby("Model Type"):
    plt.scatter(
        group["Num Parameters"],
        group["SELD Score"],
        label=model_type,
        color=colors.get(model_type, "gray"),
        s=60,
        alpha=0.8
    )

plt.xlabel("Number of Parameters")
plt.ylabel("SELD Score")
#plt.xlim(0, 8e6)
plt.title("SELD Score vs Number of Parameters")
plt.legend(title="Model Type")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("seld_vs_parameters.png", dpi=300)