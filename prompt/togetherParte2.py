import pandas as pd
import time
import logging
import os
from together import Together
from datetime import datetime

# Configuración de logging con path absoluto
log_file = os.path.abspath('together2Navil.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Inicializar cliente de Together AI
client = Together(api_key="")


def get_system_prompt(change_level):
    """Genera un prompt de sistema detallado según el nivel de cambio."""
    base_prompt = """You are an expert in image-to-image modification with Stable Diffusion XL.
Create a prompt that produces a {change_level} level of change compared to the original image.
Focus on creating a high-quality, detailed prompt for SDXL Refiner 1.0.
Do not include references to the original artist, painting name, or description in the output."""

    level_prompts = {

    "moderate": """
For this MODERATE modification:
- Keep the main composition and subject recognizable
- Transform color schemes, lighting conditions, or artistic techniques
- Add or modify secondary elements while preserving primary subjects
- Create a clear visual difference while maintaining the artwork's essence
- Consider time of day changes, season shifts, or stylistic reinterpretations

Keywords to consider: transform, shift, reinterpret, reimagine, alternative take, artistic variation""",
    
    "radical": """
For this RADICAL modification:
- Completely transform the artistic style, era, or medium
- Dramatically alter color palette, composition, or perspective
- Recontextualize the subject matter in a boldly different setting
- Create a new artistic vision that only conceptually relates to the original
- Consider genre shifts, opposing aesthetics, or unexpected conceptual fusions

IMPORTANT: Each time you create a radical transformation, choose a DIFFERENT approach. AVOID always defaulting to futuristic/cyberpunk themes. Instead, randomly select from these diverse transformation types:

1. Historical transposition (ancient, medieval, renaissance, baroque, victorian, etc.)
2. Cultural reinterpretation (different cultural aesthetics and traditions)
3. Natural/organic reimagining (biological, ecological, geological transformations)
4. Abstract/surreal deconstruction (dreamlike, symbolic, psychological)
5. Material/medium transformation (sculpture, mosaic, textile, mixed media)
6. Emotional/psychological reframing (express entirely different emotions)
7. Mythological/fantastical reinvention (folklore, fairy tales, legends)
8. Scientific/diagrammatic visualization (anatomical, astronomical, mathematical)
9. Technological reimagining (only occasionally use cyberpunk/futuristic)
10. Miniature/gigantic scale shifts (micro or macro perspectives)

For each radical prompt, SELECT ONLY ONE of these approaches rather than combining multiple.

Keywords to consider: revolutionize, transpose, transmute, overhaul, profound transformation, reimagined universe"""
    }
    
    return base_prompt.format(change_level=change_level) + level_prompts[change_level] + "\n\nCreate a detailed, vibrant prompt with descriptive adjectives. MAXIMUM 75 WORDS."

def generate_prompt(row, row_idx):
    """Genera un prompt para Stable Diffusion usando el modelo LLM de Together AI."""
    change_level = row['category'].lower()
    system_prompt = get_system_prompt(change_level)
    
    # Información contextual de la pintura
    art_context = f"""Genre: {row['genre']}
    Artist: {row['artist']}
    Title: {row['painting_name']}
    Description: {row['description']}
    Change level: {change_level}

    Generate a prompt for Stable Diffusion XL Refiner 1.0 to create a variation of this artwork. 
    IMPORTANT: Provide ONLY the prompt text without any introduction or explanation.
    ALWAYS end your prompt with quotation marks even if you reach the token limit."""
    
    # Construir el mensaje completo para Together AI
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": art_context}
    ]
    
    try:
        # Enviar la solicitud a Together AI
        logging.info(f"Enviando solicitud para fila {row_idx}: {row['file_name']}")
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
            messages=messages,
            max_tokens=200  
        )
        
        # Obtener el prompt generado y limpiarlo de posibles formatos adicionales
        generated_prompt = response.choices[0].message.content.strip()
        
        # Eliminar posibles introducciones y comillas
        if ":" in generated_prompt:
            generated_prompt = generated_prompt.split(":", 1)[1].strip()
        
        # Eliminar comillas si están presentes al inicio y final
        generated_prompt = generated_prompt.strip('"\'')
        
        # Asegurar que el texto esté en una sola línea
        generated_prompt = generated_prompt.replace('\n', ' ').replace('\r', '')
        
        # Envolver el prompt en comillas dobles
        generated_prompt = f'"{generated_prompt}"'
        
        # Mejorar el logging para incluir el prompt generado
        logging.info(f"Prompt generado para fila {row['file_name']}: {generated_prompt}")
        
        return generated_prompt
    except Exception as e:
        logging.error(f"Error en fila {row['file_name']}: {str(e)}")
        return f'ERROR: "{str(e)}"'

def process_csv(file_path, output_file, batch_size=50, delay=1):
    """Procesa un archivo CSV para generar prompts y guardar los resultados en lotes."""
    # Convertir a rutas absolutas
    file_path = os.path.abspath(file_path)
    output_file = os.path.abspath(output_file)
    
    # Crear directorio para resultados si no existe
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Leer el archivo CSV original
    df = pd.read_csv(file_path)
    total_rows = len(df)
    
    logging.info(f"Iniciando procesamiento de {total_rows} filas desde {file_path}")
    print(f"Iniciando procesamiento de {total_rows} filas desde {file_path}")
    
    # Verificar si ya existe un archivo de salida para continuar el proceso
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        processed_rows = len(existing_df)
        logging.info(f"Archivo de salida existente con {processed_rows} filas procesadas")
        print(f"Archivo de salida existente con {processed_rows} filas procesadas")
        
        # Verificar si todas las filas han sido procesadas
        if processed_rows >= total_rows:
            logging.info("Todas las filas ya han sido procesadas. No se requiere más procesamiento.")
            print("Todas las filas ya han sido procesadas. No se requiere más procesamiento.")
            return
        
        # Continuar desde donde se dejó
        df = df.iloc[processed_rows:]
        start_idx = processed_rows
    else:
        # Iniciar desde cero
        existing_df = pd.DataFrame()  # Crear DataFrame vacío en lugar de None
        start_idx = 0
    
    results = []
    current_batch = 0
    last_save_time = time.time()
    
    for idx, row in df.iterrows():
        try:
            real_idx = start_idx + (idx - df.index[0])
            prompt = generate_prompt(row, real_idx)
            
            # Crear un diccionario con los datos originales y el prompt generado
            row_data = row.to_dict()
            row_data['generated_prompt'] = prompt
            results.append(row_data)
            
            # Mostrar el prompt generado
            print(f"Row {real_idx}: {prompt}")
            
            # Guardar resultados periódicamente según batch_size o tiempo (cada 10 minutos)
            current_batch += 1
            current_time = time.time()
            time_elapsed = current_time - last_save_time
            
            if current_batch >= batch_size or time_elapsed > 600:  # 600 segundos = 10 minutos
                # Genera un timestamp único para el backup
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Guardar resultados
                save_results(results, existing_df, output_file, timestamp)
                
                # Actualizar existing_df con los nuevos resultados
                new_df = pd.DataFrame(results)
                existing_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                logging.info(f"Guardados {len(results)} resultados en lote (filas {real_idx-current_batch+1} a {real_idx})")
                print(f"Guardados {len(results)} resultados en lote (filas {real_idx-current_batch+1} a {real_idx})")
                
                results = []  # Reiniciar para el próximo lote
                current_batch = 0
                last_save_time = current_time
            
            # Respetar el límite de RPM
            time.sleep(delay)
            
        except Exception as e:
            error_msg = f"Error general al procesar la fila {real_idx}: {str(e)}"
            logging.error(error_msg)
            print(error_msg)
            
            # Agregar fila con error
            row_data = row.to_dict()
            row_data['generated_prompt'] = f"ERROR: {str(e)}"
            results.append(row_data)
            
            # Si ocurre un error, guardar lo que tenemos hasta ahora
            if len(results) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_results(results, existing_df, output_file, timestamp)
                
                # Actualizar existing_df
                new_df = pd.DataFrame(results)
                existing_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                logging.info(f"Guardados {len(results)} resultados después de error")
                print(f"Guardados {len(results)} resultados después de error")
                results = []
                current_batch = 0
                last_save_time = time.time()
    
    # Guardar cualquier resultado restante
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(results, existing_df, output_file, timestamp)
        logging.info(f"Guardados {len(results)} resultados finales")
        print(f"Guardados {len(results)} resultados finales")
    
    logging.info(f"Procesamiento completo. Resultados guardados en {output_file}")
    print(f"Procesamiento completo. Resultados guardados en {output_file}")

def save_results(new_results, existing_df, output_file, timestamp):
    """Guarda los resultados en el archivo CSV, concatenando con datos existentes."""
    if not new_results:
        logging.info("No hay nuevos resultados para guardar")
        return
        
    try:
        # Convertir resultados a DataFrame
        new_df = pd.DataFrame(new_results)
        
        # Concatenar con datos existentes
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Guardar el DataFrame combinado
        combined_df.to_csv(output_file, index=False)
        logging.info(f"Guardado principal completado: {output_file}")
        
        # También guardar una copia de respaldo con timestamp
        backup_dir = os.path.join(os.path.dirname(output_file), "backups")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            
        backup_file = os.path.join(backup_dir, f"{os.path.basename(output_file).split('.')[0]}_{timestamp}.csv")
        combined_df.to_csv(backup_file, index=False)
        logging.info(f"Backup completado: {backup_file}")
    except Exception as e:
        logging.error(f"Error al guardar resultados: {str(e)}")
        print(f"Error al guardar resultados: {str(e)}")

# Ejemplo de uso
if __name__ == "__main__":
    # Rutas absolutas para mayor seguridad
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, "parte_2.csv") 
    output_dir = os.path.join(base_dir, "together2")
    
    # Asegurar que el directorio de salida exista
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_csv = os.path.join(output_dir, "together2Navil.csv")
    
    # Reducir el tamaño del lote y agregar más puntos de guardado
    process_csv(input_csv, output_csv, batch_size=500, delay=1)