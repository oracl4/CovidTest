# Covid X-Ray Experimental Test

CovidTest is an experimental covid testing that use X-Ray images to predict the possibility of Covid-19 suspect. This project is created based of the ResNet152V2 model with modification and trained with the [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database/data).

![Example](https://github.com/oracl4/CovidTest/blob/main/example.png?raw=true)

### Installation

Download the model from this link : [Base Model](https://bit.ly/Covid19ModelMY)

Place the model inside the project folder.

Install the dependencies in requirements.txt and start the Streamlit server.

```sh
$ cd CovidTest
$ pip install -r requirements.txt
$ streamlit run CovidTest.py
```

License
----

MIT
