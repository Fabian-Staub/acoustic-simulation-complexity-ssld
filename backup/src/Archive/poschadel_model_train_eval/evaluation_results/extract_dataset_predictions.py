import pandas as pd

input_csv = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_drr_R2000_P8_S3_Order3_24kHz_shoeBox_SH_ISMOrder12.csv"
output_csv = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/poschadel/model/checkpoints/evaluation_results/predictions/predictions_arni_dataset.csv"

first_chunk = True
total_rows = 0

for chunk in pd.read_csv(input_csv, chunksize=100_000):
    filtered = chunk[chunk["prediction_dataset"] == "arni-dataset"]

    if not filtered.empty:
        filtered.to_csv(
            output_csv,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        total_rows += len(filtered)
        first_chunk = False

print(f"Saved {total_rows} rows to {output_csv}")