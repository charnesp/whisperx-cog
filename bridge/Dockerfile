FROM python:3.9-slim

WORKDIR /scripts

COPY bridge.py openai_compat.py ./

EXPOSE 8080

CMD ["python", "-u", "bridge.py"]
