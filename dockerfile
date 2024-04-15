FROM python:3.12.1
WORKDIR /app
COPY  main.py .
RUN pip install sklearn
EXPOSE 5000 
CMD ["python","model.py"]