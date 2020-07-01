FROM python:3.7.4

COPY . /usr/app/

EXPOSE 8501

WORKDIR /usr/app/

RUN pip install -r requirements.txt

CMD streamlit run WebApp.py