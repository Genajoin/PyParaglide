#latest-py3
FROM tensorflow/tensorflow:1.15.0-py3

ARG USERNAME=paraglidable
RUN useradd -m $USERNAME
ENV HOME /home/$USERNAME

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get install -y --no-install-recommends net-tools libopenjp2-7-dev libpng-dev libpng16-16 vim sass wget qt5-default pdf2svg apache2 php libapache2-mod-php nodejs npm ibgeos-3.6.2 libgeos-dev unzip

# grib
RUN apt-get -y update && apt-get install -y --no-install-recommends libgrib-api-dev libgrib-api0
RUN apt-get -y update && apt-get install -y --no-install-recommends libeccodes-dev libeccodes0

# tex2svg
#RUN npm install -y --no-install-recommends --global mathjax-node-cli
RUN npm install -y --no-install-recommends --global mathjax@3

# Add sudo support
RUN apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME


USER $USERNAME

RUN sudo -H pip install glob2 tqdm colorama requests pandas matplotlib pillow numba pyproj jupyter
RUN sudo -H pip install pygrib
#RUN sudo -H pip install https://github.com/matplotlib/basemap/archive/master.zip
RUN sudo -H pip install basemap

# IGC parsing dependencies (for parse_igc_with_libs.py)
RUN sudo -H pip install libigc aerofiles numpy scipy

# Database connectivity for PKL dataset builder
RUN sudo -H pip install 'psycopg[binary]'

