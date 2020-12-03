FROM python:3.8
WORKDIR /app
ADD . /app
ENV TZ Asia/Shanghai
RUN sed -i 's#http://deb.debian.org#http://mirrors.163.com#g' /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y iptables wondershaper iperf3 ntp ntpdate
RUN service ntp start
RUN echo "server 40.253.65.179 prefer" >> /etc/ntp.conf
RUN pip install pip -U
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements.txt
CMD ["python", "scheds.py", "40.253.65.179:2181", "40.253.65.179:9092"]