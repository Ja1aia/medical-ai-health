FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN python -m spacy download en_core_web_sm

COPY . .

RUN chmod +x scripts/startup.sh

EXPOSE 8080

CMD ["/bin/bash", "scripts/startup.sh"]