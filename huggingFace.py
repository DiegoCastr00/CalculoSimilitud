from huggingface_hub import HfApi

api = HfApi()
token = ""
repo_id = "Dant33/Wikiart"
repo_type = "dataset" 

api.upload_file(
    path_or_fileobj="wikiart_dataset.tar",
    path_in_repo="wikiart_dataset.tar",
    repo_id=repo_id,
    token=token,
    repo_type=repo_type,
)

print("Archivo subido exitosamente a Hugging Face.")
