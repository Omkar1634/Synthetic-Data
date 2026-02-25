import torch
import sys
from collections import OrderedDict

def inspect_pt_file(path):
    print(f"\n🔍 Loading: {path}\n")

    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        print("❌ Error loading file:", e)
        return

    print("=" * 60)
    print("📌 TYPE OF OBJECT STORED:")
    print("=" * 60)
    print(type(data))

    print("\n" + "=" * 60)
    print("📌 BASIC CONTENT OVERVIEW:")
    print("=" * 60)

    # Case 1: Full model
    if isinstance(data, torch.nn.Module):
        print("✅ This file contains a FULL MODEL\n")

        print("📐 Model Architecture:")
        print(data)

        print("\n📦 Total Parameters:")
        total_params = sum(p.numel() for p in data.parameters())
        trainable_params = sum(p.numel() for p in data.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

        print("\n🧱 Layer Details:")
        for name, module in data.named_modules():
            print(f"Layer: {name} -> {type(module)}")

        print("\n📊 Parameter Shapes:")
        for name, param in data.named_parameters():
            print(f"{name}: {tuple(param.shape)} | requires_grad={param.requires_grad}")

    # Case 2: State dict
    elif isinstance(data, (dict, OrderedDict)):

        print("✅ This file contains a DICTIONARY / CHECKPOINT\n")
        print("🔑 Keys inside file:")
        for key in data.keys():
            print(f"- {key}")

        # If pure state_dict
        if all(isinstance(v, torch.Tensor) for v in data.values()):
            print("\n📦 This appears to be a pure STATE_DICT (weights only)\n")

            total_params = 0
            print("📊 Parameter Shapes:")
            for name, tensor in data.items():
                print(f"{name}: {tuple(tensor.shape)}")
                total_params += tensor.numel()

            print(f"\nTotal parameters (from state_dict): {total_params}")

        # If checkpoint
        else:
            print("\n📂 This appears to be a CHECKPOINT file\n")

            if "epoch" in data:
                print(f"📅 Epoch: {data['epoch']}")

            if "loss" in data:
                print(f"📉 Loss: {data['loss']}")

            if "model_state_dict" in data:
                print("\n📦 Model State Dict Found")
                model_state = data["model_state_dict"]

                total_params = 0
                for name, tensor in model_state.items():
                    print(f"{name}: {tuple(tensor.shape)}")
                    total_params += tensor.numel()

                print(f"\nTotal parameters: {total_params}")

            if "optimizer_state_dict" in data:
                print("\n⚙️ Optimizer state found")

    else:
        print("⚠️ Unknown file structure")
        print(data)


if __name__ == "__main__":
    model_path = r"D:\Github\PhD Code\Synthetic Data\NN\BioSkin_trained_models\BioSkinRGB.pt"
    inspect_pt_file(model_path)