from PIL import Image
import io

from src.inference import predict_from_bytes

# 더미 이미지 생성
img = Image.new("RGB", (300, 300), color="red")
buf = io.BytesIO()
img.save(buf, format="JPEG")
image_bytes = buf.getvalue()

result = predict_from_bytes(image_bytes)
print(result)
