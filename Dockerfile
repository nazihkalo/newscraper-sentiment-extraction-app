FROM python:3.7

WORKDIR /app/

COPY streamrequirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

RUN python -m nltk.downloader punkt

EXPOSE 8080

COPY . /app

CMD streamlit run --server.port 8080 --server.enableCORS false sentimentapp.py
# ENTRYPOINT [ "streamlit", "run" ]

# CMD [ "sentimentapp.py" ]
