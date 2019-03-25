import base64
import io
import json

import requests
from PIL import Image

from flask import request
from flask_api import FlaskAPI
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms


app = FlaskAPI("image_classification")


@app.route("/predict", methods=['POST'])
def predict_image_class():

    link_to_image = request.data.get('link')
    text_img = request.data.get('text_img')

    if link_to_image:
        image_content = requests.get(link_to_image)
    elif text_img:
        image_content = base64.b64decode(text_img)
    else:
        return {'error': 'Image is not provided'}, 400

    img_pil = Image.open(io.BytesIO(image_content))
    vgg_model = models.vgg16(pretrained=True)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)
    fc_out = vgg_model(img_variable)
    predicted_idx = fc_out.data.argmax()

    class_idx = json.load(open("imagenet1000.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    label = idx2label[predicted_idx]

    return {'imagenet_idx': int(predicted_idx), 'imagenet_label': label}


if __name__ == "__main__":
    app.run(debug=True)



