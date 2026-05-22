[Vehicle Tracking Dataset](https://zenodo.org/records/18195750)

```bash
cd MultiTracking
```

Create virtual environments
```bash
python3 -m venv .venv
```

Open virtual environments
```bash
source .venv/bin/activate
```

Checking python version and update pip
```bash
python -m pip install --upgrade pip setuptools wheel
```

Install package
```bash
pip install -r requirements.txt
```

Project using 2 models
```text
models/yolov5n.pt
models/ckpt.t7
```

Run
```bash
python app.py
```
And observe on the website interface in localhost, follow the link
```bash
http://127.0.0.1:5000
```
or in other devices on the same Wi-Fi/LAN network, follow the link:
```bash
http://192.168.0.113:5000
```

Exist virtual environments
```bash
deactivate
```