FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install pillow
RUN pip install image4layer
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt Pillow
EXPOSE 5000
ENV FLASK_APP=gen_drawing.py
CMD ["python", "gen_drawing.py"]
