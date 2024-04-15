FROM python:3.12.1
WORKDIR /app
COPY  main_file.py .
RUN pip install pandas
RUN pip install numpy
RUN pip install -U scikit-learn
COPY framingham.csv /app
EXPOSE 5000 
CMD ["python","main_file.py"]
