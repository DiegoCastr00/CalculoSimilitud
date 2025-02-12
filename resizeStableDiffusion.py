import os
from PIL import Image

class ImageResizer:
    def __init__(self, input_dir, output_dir, target_size=(1024, 1024), resample=Image.LANCZOS):
        """
        Inicializa el redimensionador de imágenes.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.resample = resample
        os.makedirs(output_dir, exist_ok=True)
    
    def resize_image(self, image_path, output_path):
        """
        Redimensiona una imagen individual.
        """
        try:
            img = Image.open(image_path)
            resized_img = img.resize(self.target_size, resample=self.resample)
            resized_img.save(output_path)
            # print(f"Imagen redimensionada guardada: {output_path}")
        except Exception as e:
            print(f"Error al procesar la imagen {image_path}: {e}")
    
    def resize_all_images(self):
        """
        Redimensiona todas las imágenes en el directorio de entrada.
        """
        for filename in os.listdir(self.input_dir):
            input_path = os.path.join(self.input_dir, filename)
            output_path = os.path.join(self.output_dir, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                self.resize_image(input_path, output_path)
            else:
                print(f"Archivo ignorado (no es una imagen): {filename}")
                
if __name__ == "__main__":
    input_directory = "./imagenes"
    output_directory = "./output"
    target_size = (1024, 1024)
    resizer = ImageResizer(input_dir=input_directory, output_dir=output_directory, target_size=target_size)
    resizer.resize_all_images()