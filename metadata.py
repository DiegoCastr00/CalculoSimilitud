import os
import csv

input_folder = "imagenes/resizeSD"
output_csv_file = "metadata.csv"

with open(output_csv_file, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file_name", "label", "description"]) 
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")): 
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, input_folder)
                label = os.path.basename(root)  
                description = f"Descripción de {file}" 
                writer.writerow([relative_path, label, description])
                
print(f"Archivo {output_csv_file} creado exitosamente.")