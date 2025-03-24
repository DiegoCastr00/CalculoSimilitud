from huggingface_hub import HfApi, HfFolder, upload_file

token = ""

repo_id = "Dant33/ArtSimilarity" 
repo_type = "model"

api = HfApi()

upload_file(
    path_or_fileobj="model.pt",  
    path_in_repo="model.pt",
    repo_id=repo_id,
    token=token,
    repo_type=repo_type,
)

upload_file(
    path_or_fileobj="inference.py",
    path_in_repo="inference.py",
    repo_id=repo_id,
    token=token,
    repo_type=repo_type,
)

print("Modelo subido exitosamente a Hugging Face.")
