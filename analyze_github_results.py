import pandas as pd
import numpy as np

df = pd.read_csv("results/metrics_github.csv")
df["device"] = df["sample_id"].str.rsplit("_", n=1).str[0]

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

print("=== overall rel_l2 by N (mean +/- std over seeds, then chips) ===")
per_seed = df.groupby(["sample_id", "n_sensors", "seed"])["rel_l2"].mean().reset_index()
print(per_seed.groupby("n_sensors")["rel_l2"].agg(["mean", "std", "min", "max", "count"]))

print("\n=== by device x N ===")
print(df.groupby(["device", "n_sensors"])["rel_l2"].mean().unstack())

print("\n=== peak_temp_err (degC) by device x N ===")
print(df.groupby(["device", "n_sensors"])["peak_temp_err"].mean().unstack())

print("\n=== hotspot_loc_cells by device x N ===")
print(df.groupby(["device", "n_sensors"])["hotspot_loc_cells"].mean().unstack())

print("\n=== sensor_rmse by device x N (did it even fit its own sensors?) ===")
print(df.groupby(["device", "n_sensors"])["sensor_rmse"].mean().unstack())

print("\n=== worst 10 rows by rel_l2 ===")
print(df.nlargest(10, "rel_l2")[["sample_id", "n_sensors", "seed", "rel_l2", "peak_temp_err", "hotspot_loc_cells", "true_peak_temp"]])

print("\n=== best 10 rows by rel_l2 ===")
print(df.nsmallest(10, "rel_l2")[["sample_id", "n_sensors", "seed", "rel_l2", "peak_temp_err", "hotspot_loc_cells", "true_peak_temp"]])

print("\n=== mean train_seconds per solve ===")
print(df["train_seconds"].describe())
