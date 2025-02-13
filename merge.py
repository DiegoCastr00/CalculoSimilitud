import pandas as pd

dataset = pd.read_csv("dataset.csv")
metadata = pd.read_csv("metadata.csv")

metadata["original_filename"] = metadata["file_name"].str.replace("_resize1024", "", regex=False)

merged = metadata.merge(dataset, left_on="original_filename", right_on="filename", how="left")

final_metadata = merged[[ "file_name", "genre", "artist","description", "phash",  "subset"]]

final_metadata.to_csv("metadata_updated.csv", index=False)

print("Proceso completado. Archivo guardado como 'metadata_updated.csv'")
