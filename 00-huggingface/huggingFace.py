from huggingface_hub import HfApi

api = HfApi()
token = ""
repo_id = "Dant33/WikiArt-81K-BLIP_2-768x768"
repo_type = "dataset" 

api.upload_file(
    path_or_fileobj="dataset.tar.gz",
    path_in_repo="dataset.tar.gz",
    repo_id=repo_id,
    token=token,
    repo_type=repo_type,
)

print("Archivo subido exitosamente a Hugging Face.")

