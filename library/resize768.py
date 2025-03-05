import os
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

class ImageResizerWithCSV:
    def __init__(self, csv_path, input_base_folder, output_base_folder, target_size=(768, 768)):
        self.csv_path = csv_path
        self.input_base_folder = input_base_folder
        self.output_base_folder = output_base_folder
        self.target_size = target_size
        os.makedirs(output_base_folder, exist_ok=True)

    def resize_with_padding(self, image_path):
        try:
            # Abre la imagen
            img = Image.open(image_path)
            
            # Calcula el ratio para mantener la proporción
            ratio = min(self.target_size[0] / img.width, self.target_size[1] / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            
            # Redimensiona la imagen
            img_resized = img.resize(new_size, resample=Image.LANCZOS)
            
            # Crea una nueva imagen con fondo negro
            new_img = Image.new("RGB", self.target_size, (0, 0, 0))  # Negro
            
            # Calcula la posición para centrar la imagen
            paste_position = ((self.target_size[0] - new_size[0]) // 2, (self.target_size[1] - new_size[1]) // 2)
            
            # Pega la imagen redimensionada en el centro
            new_img.paste(img_resized, paste_position)
            
            return new_img
        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
            return None

    def process_image(self, row):
        file_name = row["file_name"]
        processed_prompt = row["processed_prompt"]

        # Extrae el nombre original sin "_resize1024"
        original_file_name = file_name.replace("_resize1024", "")
        input_path = os.path.join(self.input_base_folder, original_file_name)

        # Si no encuentra la imagen original, busca la versión "_resize1024" en el directorio alternativo
        if not os.path.exists(input_path):
            print(f"No se encontró la imagen original: {input_path}")
            fallback_path = os.path.join("imagenes/resizeSD/", file_name)  # Usa el nombre con "_resize1024"
            if os.path.exists(fallback_path):
                input_path = fallback_path
                print(f"Usando fallback: {fallback_path}")
            else:
                print(f"Error: No se encontró ninguna versión de la imagen: {file_name}")
                return None

        # Define la salida
        output_dir = os.path.dirname(os.path.join(self.output_base_folder, original_file_name))
        os.makedirs(output_dir, exist_ok=True)  # Crea subdirectorios si no existen
        output_file_name = os.path.splitext(original_file_name)[0] + "_resize768.jpg"
        output_path = os.path.join(self.output_base_folder, output_file_name)

        # Verifica si la imagen ya fue procesada
        if os.path.exists(output_path):
            print(f"Saltando {output_file_name} (ya procesada)")
            return {"file_name": output_file_name, "processed_prompt": processed_prompt}

        # Redimensiona y guarda
        resized_image = self.resize_with_padding(input_path)
        if resized_image:
            resized_image.save(output_path, format="JPEG", quality=95)  # Calidad alta
            print(f"Procesada: {output_file_name}")
            return {"file_name": output_file_name, "processed_prompt": processed_prompt}
        else:
            return None

    def process_all_images(self, max_workers=8):
        df = pd.read_csv(self.csv_path)

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_image, row) for _, row in df.iterrows()]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        new_df = pd.DataFrame(results)
        new_csv_path = os.path.splitext(self.csv_path)[0] + "_resize768.csv"
        new_df.to_csv(new_csv_path, index=False)
        print(f"Nuevo CSV generado: {new_csv_path}")

csv_path = "compressPromptLite.csv"
input_base_folder = "imagenes/extracted_images/"
output_base_folder = "imagenes/resize768/"

resizer = ImageResizerWithCSV(csv_path, input_base_folder, output_base_folder)
resizer.process_all_images(max_workers=8)