import tarfile
import os

# Rutas de entrada
input_folder = "imagenes/resize768"  # Carpeta con las imágenes
metadata_file_1 = "data.csv"            # Primer archivo CSV con los metadatos
# metadata_file_2 = "metadata.csv"     # Segundo archivo CSV adicional
output_tar_file = "dataset.tar.gz"      # Nombre del archivo comprimido

# Crear el archivo .tar.gz
with tarfile.open(output_tar_file, "w:gz") as tar:
    # Añadir las imágenes
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)
            # Calcular la ruta relativa dentro de la carpeta "images/"
            arcname = os.path.relpath(file_path, input_folder)
            arcname = os.path.join("images", arcname.replace("\\", "/"))  # Asegurar diagonales correctas
            tar.add(file_path, arcname=arcname)
    
    # Añadir el primer archivo CSV
    if os.path.exists(metadata_file_1):
        tar.add(metadata_file_1, arcname="data.csv")
    else:
        
        raise FileNotFoundError(f"El archivo {metadata_file_1} no fue encontrado.")
    
    # # Añadir el segundo archivo CSV
    # if os.path.exists(metadata_file_2):
    #     tar.add(metadata_file_2, arcname="additional_data.csv")
    # else:
    #     print(f"Advertencia: El archivo {metadata_file_2} no fue encontrado. No se agregará al archivo comprimido.")

print(f"Archivo {output_tar_file} creado exitosamente.")