# Try to load a model from a specific path using timm.create_model to check if resnet18
import timm

checkpoint_path = "data/0005.pth.tar"
target_model = timm.models.create_model("resnet18", checkpoint_path=checkpoint_path, pretrained=False)