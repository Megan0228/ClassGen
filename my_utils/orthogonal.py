

yx = all_image_features.t() @ all_text_features
u, s, v = torch.svd(yx)
w = u @ v.T