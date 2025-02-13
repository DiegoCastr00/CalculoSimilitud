import tarfile
import os

input_folder = "imagenes/resizeSD"
metadata_file = "metadata.csv"
output_tar_file = "wikiart_dataset.tar"

with tarfile.open(output_tar_file, "w") as tar:
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, input_folder)
            tar.add(file_path, arcname=f"images/{arcname}")
    
    tar.add(metadata_file, arcname="metadata.csv")

print(f"Archivo {output_tar_file} creado exitosamente.")