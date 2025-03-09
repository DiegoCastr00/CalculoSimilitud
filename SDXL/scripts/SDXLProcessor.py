import os
import pandas as pd
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
import gc
from diffusers import AutoencoderTiny

class SDXLImageProcessor:
    def __init__(self, 
                 gpu_id=0,
                 csv_path="datos.csv",
                 input_dir="imagenes/resize768",
                 output_dir="imagenes/output",
                 num_inference_steps=25,
                 strength=0.4,
                 guidance_scale=7.5,
                 batch_size=6):

        self.gpu_id = gpu_id
        self.csv_path = csv_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.pipe = None
        self.batch_size = batch_size
        
        # No configuramos CUDA_VISIBLE_DEVICES aquí para permitir usar distintas GPUs
        print(f"Inicializando procesador con GPU ID: {self.gpu_id}")
    
    def setup_pipeline(self):
        """Configura el pipeline SDXL para la GPU especificada"""
        # Seleccionamos la GPU específica directamente en PyTorch
        device = torch.device(f"cuda:{self.gpu_id}")
        print(f"Dispositivo seleccionado: {device}")
        print(f"GPU disponible: {torch.cuda.get_device_name(self.gpu_id)}")
        print(f"Memoria total: {torch.cuda.get_device_properties(self.gpu_id).total_memory / 1e9:.2f} GB")
        vae = AutoencoderTiny.from_pretrained(
                'madebyollin/taesdxl',
                use_safetensors=True,
                torch_dtype=torch.float16,  
            ).to(device)  # Usa el device específico, no hardcoded 'cuda'
    
        
        # Cargar el modelo optimizado
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            vae=vae,

        )
        
        # Mover el modelo al dispositivo CUDA específico
        self.pipe = self.pipe.to(device)
        
        # Optimizaciones de memoria y rendimiento
        self.pipe.enable_model_cpu_offload(gpu_id=self.gpu_id)  # Especificar GPU ID
        self.pipe.enable_vae_slicing()  # Reduce el uso de memoria del VAE
        self.pipe.enable_xformers_memory_efficient_attention()  # Usa xformers para atención eficiente
        
        print(f"Pipeline configurado en GPU {self.gpu_id}")
    
    def process_images_batch(self):
        """Procesa las imágenes en lotes para mayor velocidad"""
        if self.pipe is None:
            self.setup_pipeline()
        
        os.makedirs(self.output_dir, exist_ok=True)
        df = pd.read_csv(self.csv_path)
        
        print(f"Procesando {len(df)} imágenes del archivo {self.csv_path} en la GPU {self.gpu_id}")
        
        # Procesar en lotes
        batch_images = []
        batch_prompts = []
        batch_paths = []
        
        for idx, row in df.iterrows():
            try:
                file_path = row['file_name']
                prompt = row['processed_prompt']
                image_path = os.path.join(self.input_dir, file_path)
                
                if not os.path.exists(image_path):
                    print(f"Advertencia: Imagen no encontrada - {image_path}")
                    continue
                
                # Agregar al lote actual
                init_image = Image.open(image_path).convert("RGB")
                batch_images.append(init_image)
                batch_prompts.append(prompt)
                batch_paths.append(file_path)
                
                # Procesar cuando el lote esté completo o al final
                if len(batch_images) == self.batch_size or idx == df.index[-1]:
                    # Asegurar que estamos en la GPU correcta
                    device = torch.device(f"cuda:{self.gpu_id}")
                    
                    # Generar imágenes en lote
                    outputs = self.pipe(
                        prompt=batch_prompts,
                        image=batch_images,
                        num_inference_steps=self.num_inference_steps,
                        strength=self.strength,
                        guidance_scale=self.guidance_scale,
                    )
                    
                    # Guardar resultados
                    for i, output_img in enumerate(outputs.images):
                        # Obtener la ruta de salida, manteniendo la estructura de carpetas
                        relative_path = os.path.dirname(batch_paths[i])
                        base_filename = os.path.splitext(os.path.basename(batch_paths[i]))[0]
                        
                        # Crear directorio de salida si no existe
                        output_dir_path = os.path.join(self.output_dir, relative_path)
                        os.makedirs(output_dir_path, exist_ok=True)
                        
                        # Guardar la imagen generada
                        output_filename = f"{base_filename}_generated.jpg"
                        output_path = os.path.join(output_dir_path, output_filename)
                        output_img.save(output_path)
                        
                        # Imprimir mensaje de imagen procesada
                        print(f"Imagen procesada y guardada: {output_path}")
                    
                    # Limpiar el lote
                    batch_images = []
                    batch_prompts = []
                    batch_paths = []
                    
                    # Limpieza periódica de memoria específica para esta GPU
                    if idx % 20 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
            
            except Exception as e:
                print(f"Error procesando {file_path}: {str(e)}")
                
                # Reiniciar el lote en caso de error
                batch_images = []
                batch_prompts = []
                batch_paths = []
                
                # Intentar limpiar memoria en caso de error
                torch.cuda.empty_cache()
                gc.collect()
        
        self._cleanup()
        print(f"Procesamiento completado en GPU {self.gpu_id}")
    
    def _cleanup(self):
        """Libera recursos y memoria"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()