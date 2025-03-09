import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer
import re
import os
import logging
from tqdm import tqdm
import torch.multiprocessing as mp

def process_on_gpu(args):

    batch_df, processor, gpu_id = args
    with torch.cuda.device(gpu_id):
        return processor.process_batch(batch_df)

class PromptSummary:
    def __init__(self, csv_path, output_path=None, log_file=None):
        self.csv_path = csv_path
        self.output_path = output_path or os.path.join(
            os.path.dirname(csv_path),
            f"processed_{os.path.basename(csv_path)}"
        )

        self.log_file = log_file or os.path.join(
            os.path.dirname(csv_path),
            f"log_{os.path.basename(csv_path).split('.')[0]}.log"
        )
        self._setup_logging()

        self.num_gpus = torch.cuda.device_count()
        self.logger.info(f"GPUs detectadas: {self.num_gpus}")

        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if self.num_gpus > 0 else -1,
            batch_size=16
        )

        self.data = pd.read_csv(csv_path)
        self.logger.info(f"Datos cargados: {len(self.data)} filas")

        required_cols = ['file_name', 'prompt']
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            self.logger.error(f"Columnas faltantes en el CSV: {missing}")
            raise ValueError(f"Columnas faltantes en el CSV: {missing}")

    def _setup_logging(self):
        self.logger = logging.getLogger('PromptProcessor')
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        self.logger.info(f"Iniciando PromptProcessor para {self.csv_path}")
        self.logger.info(f"Logs guardados en: {self.log_file}")

    def preprocess_prompt(self, prompt):
        cleaned = re.sub(r'Transform .+? into ', '', prompt)
        if not cleaned.endswith('.'):
            cleaned += '.'
        return cleaned

    def check_token_length(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def process_batch(self, batch_df, target_length=60):
        result_df = batch_df.copy()
        result_df['cleaned_prompt'] = result_df['prompt'].apply(self.preprocess_prompt)
        result_df['token_length'] = result_df['cleaned_prompt'].apply(self.check_token_length)

        if (result_df['token_length'] > 77).any():
            prompts_to_summarize = result_df.loc[result_df['token_length'] > 77, 'cleaned_prompt'].tolist()

            summarized = self.summarizer(
                prompts_to_summarize,
                max_length=target_length,
                min_length=40,
                do_sample=False,
                truncation=True,
            )

            summarized_texts = [s['summary_text'] for s in summarized]
            result_df.loc[result_df['token_length'] > 77, 'processed_prompt'] = summarized_texts
            result_df.loc[result_df['token_length'] > 77, 'final_length'] = (
                result_df.loc[result_df['token_length'] > 77, 'processed_prompt'].apply(self.check_token_length)
            )

            for _, row in result_df[result_df['token_length'] > 77].iterrows():
                self.logger.info(f"Procesado {row['file_name']}: '{row['processed_prompt']}'")

        result_df.loc[result_df['token_length'] <= 77, 'processed_prompt'] = result_df['cleaned_prompt']
        result_df.loc[result_df['token_length'] <= 77, 'final_length'] = result_df['token_length']

        for _, row in result_df[result_df['token_length'] <= 77].iterrows():
            self.logger.debug(f"Ya en límite {row['file_name']}: {row['token_length']} tokens (mantenido)")

        return result_df[['file_name', 'prompt', 'processed_prompt', 'token_length', 'final_length']]

    def process_all(self, batch_size=32):
        data_batches = [self.data[i:i+batch_size] for i in range(0, len(self.data), batch_size)]
        self.logger.info(f"Dividiendo datos en {len(data_batches)} lotes de {batch_size}")

        all_results = []

        # Create arguments for the process_on_gpu function
        gpu_assignments = [(batch, self, i % max(1, self.num_gpus)) for i, batch in enumerate(data_batches)]

        with tqdm(total=len(data_batches), desc="Procesando lotes") as pbar:
            with mp.Pool(processes=max(1, self.num_gpus)) as pool:
                for result in pool.imap_unordered(process_on_gpu, gpu_assignments):
                    all_results.append(result)
                    pbar.update(1)
                    self.logger.debug(f"Lote completado ({pbar.n}/{len(data_batches)})")

        final_df = pd.concat(all_results)
        final_df = final_df.sort_index()

        return final_df

    def run(self, batch_size=32):
        self.logger.info(f"Iniciando procesamiento de {len(self.data)} prompts")

        processed_df = self.process_all(batch_size=batch_size)

        total = len(processed_df)
        over_limit_before = sum(processed_df['token_length'] > 77)
        over_limit_after = sum(processed_df['final_length'] > 77)

        self.logger.info(f"\nResumen de procesamiento:")
        self.logger.info(f"Total de prompts: {total}")
        self.logger.info(f"Prompts que excedían el límite: {over_limit_before} ({over_limit_before/total*100:.2f}%)")
        self.logger.info(f"Prompts que aún exceden el límite después de resumir: {over_limit_after} ({over_limit_after/total*100:.2f}%)")

        processed_df.to_csv(self.output_path, index=False)
        self.logger.info(f"Resultados guardados en: {self.output_path}")

        return self.output_path

if __name__ == "__main__":
    # Optional: Set start method for multiprocessing
    mp.set_start_method('spawn', force=True)  # More stable than fork for CUDA operations
    
    processor = PromptSummary(
        csv_path="datatset_prompts_lite.csv",
        output_path="summaryResultados.csv",
        log_file="summaryPromps.log"
    )
    processor.run(batch_size=64)