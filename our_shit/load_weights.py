import torch

from image_classifier import ImageClassifier


def load_weights():
    state_dict = torch.load("Digit-Classification-Pytorch/model_state.pt")

    keys = list(state_dict.keys())
    for key in keys:
        if "model" in key:
            if "7" in key:
                state_dict[key.replace("model.7", "fc_layers.1")] = state_dict[key]
                state_dict.pop(key)
                continue
            state_dict[key.replace("model", "conv_layers")] = state_dict[key]
            state_dict.pop(key)
    img_clsfr = ImageClassifier()
    img_clsfr.load_state_dict(state_dict)
    img_clsfr.eval()

    return img_clsfr
