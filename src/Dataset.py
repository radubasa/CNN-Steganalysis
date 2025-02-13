import kagglehub

# Download latest version
path = kagglehub.dataset_download("marcozuppelli/stegoimagesdataset")

print("Path to dataset files:", path)