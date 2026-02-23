from inference import load_model, process_single_image, process_folder
import torch

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder, decoder = load_model(r'D:\Github\Deep-Albedo\py_autoencoder\checkpoints\2025-12-31_18-54-39\best.pt', device)

# Process
result = process_folder(r'D:\Github\Deep-Albedo\py_autoencoder\Data\Canon_1-223052001577\raw',
    encoder,
    decoder,
    device,
    output_dir='results/',
    # show_plot=True
)
# D:\Github\Deep-Albedo\py_autoencoder\Data\250902_1854_Shraddha Front_2714f1b5-7b10-404a-8df2-c43685f23e1a\2025-09-02-18-57-06
# D:\Github\Deep-Albedo\py_autoencoder\Data\250902_1648_Omkar-front_0d34ecae-5621-408a-80f8-5f60f6216025\2025-09-02-16-48-18


