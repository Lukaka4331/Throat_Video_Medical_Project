# generate_rotation_data

產生左右旋轉20度的增強資料

# Pytorch Functional Transforms -



 ```
from torchvision.transforms import functional as F
from PIL import Image

angle=random.randrange(-20, 20)
img = Image.open(img_path).convert("RGB")
img = F.rotate(img,angle,fill=0)
F.rotate(img,angle,fill=0)
```
