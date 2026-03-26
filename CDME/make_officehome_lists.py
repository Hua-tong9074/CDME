import os

root = "./data/OfficeHome"
domains = ["Art", "Clipart", "Product", "RealWorld"]
img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

def list_dir(path):
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

# 用 Art 的类别做全局基准，确保四域 label 对齐
base_classes = list_dir(os.path.join(root, "Art"))

for d in domains:
    d_dir = os.path.join(root, d)
    save_path = os.path.join(d_dir, "image_list.txt")
    cnt = 0
    with open(save_path, "w") as f:
        for label, cls in enumerate(base_classes):
            cls_dir = os.path.join(d_dir, cls)
            if not os.path.isdir(cls_dir): 
                continue
            for name in os.listdir(cls_dir):
                ext = os.path.splitext(name)[1].lower()
                if ext in img_exts:
                    f.write(f"{cls}/{name} {label}\n")
                    cnt += 1
    print(f"[OK] wrote {save_path} ({cnt} lines)")
print("[DONE]")

