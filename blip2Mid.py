import os
import time
import pandas as pd
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device_ids = [0, 1]  
torch.cuda.set_device(device_ids[0])

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",     
    torch_dtype=torch.float16
)

model = torch.nn.DataParallel(model, device_ids=device_ids).to("cuda")

metadata_path = "metadata.csv"
data = pd.read_csv(metadata_path)

data["file_name"] = data["file_name"].str.replace("imagenes/resizeSD/", "", regex=False)

data = data.head(100)
def generate_caption(image_path):
    try:
        raw_image = Image.open("imagenes/resizeSD/" + image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
        out = model.module.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True).replace("\n", " ").replace(",", ";")
        return caption
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return "Error"

start_time = time.time()
captions = []

for i, row in data.iterrows():
    image_path = row["file_name"]
    caption = generate_caption(image_path)
    captions.append(caption)
    # print(f"Imagen {i+1}/{len(data)}: {caption}")

    if i % 100 == 0 and i != 0:
        print(f"Se han procesado {i} imágenes...")

data["blip2"] = captions
data.to_csv("metadata_blip2.csv", index=False)

end_time = time.time()
print(f"\nProcesamiento completado en {end_time - start_time:.2f} segundos.")

import gc
gc.collect()
torch.cuda.empty_cache()
