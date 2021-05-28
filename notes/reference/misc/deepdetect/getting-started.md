# Deepdetect

## Installing

* Setup Vagrant box

```
vagrant up
vagrant ssh
```

* Install deps

```
sudo apt-get install \
build-essential \
libgoogle-glog-dev \
libgflags-dev \
libeigen3-dev \
libopencv-dev \
libcppnetlib-dev \
libboost-dev \
libcurlpp-dev \
libcurl4-openssl-dev \
protobuf-compiler \
libopenblas-dev \
libhdf5-dev \
libprotobuf-dev \
libleveldb-dev \
libsnappy-dev \
liblmdb-dev \
libutfcpp-dev \
git \
cmake
```

* Compile from source

```
git clone https://github.com/beniz/deepdetect.git
cd deepdetect
mkdir build
cd build
cmake ..
make
```

* Run the server

```
cd main
./dede
```

## Installing models

Can find a list of models [here](http://www.deepdetect.com/applications/model/#lic).

In the example, I'll try downloading the [clothing](http://www.deepdetect.com/models/clothing.tar.bz2) dataset.

```
cd ~
mkdir dd-models && cd dd-models
wget http://www.deepdetect.com/models/clothing.tar.bz2
tar -xzvf //www.deepdetect.com/models/clothing.tar.bz2
```

Then, create the service like so:

```
curl -XPUT "http://localhost:8080/services/clothing" -d '{
  "mllib": "caffe",
  "description": "clothes classification",
  "type": "supervised",
  "parameters": {
    "input": {
      "connector": "image", "height": 224, "width": 224
    },
    "mllib": {
      "nclasses": 304
    }
  },
  "model": {
    "repository": "/home/vagrant/dd-models/clothing"
  }
}'
```

Now, test the service using images taken from Instagram.

```
curl -X POST "http://localhost:8080/predict" -d '{
  "service": "clothing",
  "parameters": {
    "output": {
      "best":5
    }
  },
  "data": ["https://igcdn-photos-e-a.akamaihd.net/hphotos-ak-xpf1/t51.2885-15/e35/10413268_1575933466063876_101454367_n.jpg"]
}' | python -mjson.tool
{
    "body": {
        "predictions": {
            "classes": [
                {
                    "cat": "pea jacket, peacoat",
                    "prob": 0.5521102547645569
                },
                {
                    "cat": "outerwear, overclothes",
                    "prob": 0.36573296785354614
                },
                {
                    "cat": "black",
                    "prob": 0.033303018659353256
                },
                {
                    "cat": "woman's clothing",
                    "prob": 0.013000347651541233
                },
                {
                    "cat": "greatcoat, overcoat, topcoat",
                    "last": true,
                    "prob": 0.011324126273393631
                }
            ],
            "uri": "https://igcdn-photos-e-a.akamaihd.net/hphotos-ak-xpf1/t51.2885-15/e35/10413268_1575933466063876_101454367_n.jpg"
        }
    },
    "head": {
        "method": "/predict",
        "service": "clothing",
        "time": 6749.0
    },
    "status": {
        "code": 200,
        "msg": "OK"
    }
}
```
