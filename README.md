# Deploy titanic model as API by using FastAPI

### Create virtual environment
```bash
 conda create -n fast-api-env python=3.7
```

### Activate virtual environment
```bash
 conda activate fast-api-env
```

### Install requirements packages
```bash
 pip install -r requirements.txt
```

### Start API server
```bash
 uvicorn main:app --reload
```

