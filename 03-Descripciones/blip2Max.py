import os
import time
import pandas as pd
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device_ids = [0, 1]  
torch.cuda.set_device(device_ids[0])

# Cargar modelo y procesador
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
model = torch.nn.DataParallel(model, device_ids=device_ids).to("cuda")

# Rutas de archivos
metadata_path = "metadata.csv"
output_csv = "metadata_blip2.csv"

# Cargar datos originales
data = pd.read_csv(metadata_path)
data["file_name"] = data["file_name"].str.replace("imagenes/resizeSD/", "", regex=False)

# Cargar progreso previo si existe
if os.path.exists(output_csv):
    existing_data = pd.read_csv(output_csv)
    if "blip2" in existing_data.columns:
        data = data.merge(existing_data[["file_name", "blip2"]], on="file_name", how="left")
    else:
        data["blip2"] = None
else:
    data["blip2"] = None

def generate_caption(image_path):
    try:
        raw_image = Image.open("imagenes/resizeSD/" + image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt").to("cuda")
        out = model.module.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True).replace("\n", " ").replace(",", ";")
        return caption
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return "Error"

start_time = time.time()
batch_size = 2000
processed = 0

for i, row in data.iterrows():
    if pd.notna(row["blip2"]):  # Si ya tiene descripción, saltar
        continue

    image_path = row["file_name"]
    data.at[i, "blip2"] = generate_caption(image_path)
    processed += 1

    if processed % 100 == 0:
        print(f"Se han procesado {processed} imágenes nuevas...")

    # Guardar cada 2000 imágenes
    if processed % batch_size == 0:
        data.to_csv(output_csv, index=False)
        print(f"Guardado parcial en {output_csv}")

# Guardado final
data.to_csv(output_csv, index=False)
print(f"\nProcesamiento completado y guardado en {output_csv}")

# Liberar memoria
import gc
gc.collect()
torch.cuda.empty_cache()

end_time = time.time()
print(f"\nTiempo total: {end_time - start_time:.2f} segundos.")
