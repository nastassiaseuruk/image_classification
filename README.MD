### HOW TO USE
Install requirements and run local server (flask default: `http://127.0.0.1:5000/`)
```
pip install -r requirements.txt
python ./main.py
```
Open Postman (or any other similar tool) and send POST request to `http://127.0.0.1:5000/predict` with link or text_img in json
```
{
	"link": "https://murlo.org/ru/wp-content/uploads/2014/09/%D0%95%D0%B3%D0%B8%D0%BF%D0%B5%D1%82%D1%81%D0%BA%D0%B0%D1%8F-%D0%BA%D0%BE%D1%88%D0%BA%D0%B0-%D0%BC%D0%B0%D1%83-%D1%84%D0%BE%D1%82%D0%BE.jpg"
}
```
As a result you will get json with index and label on Imagenet
```
{
    "imagenet_idx": 285,
    "imagenet_label": "Egyptian_cat"
}
```
