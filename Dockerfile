FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
