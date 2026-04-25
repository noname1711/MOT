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

Exist virtual environments
```bash
deactivate
```