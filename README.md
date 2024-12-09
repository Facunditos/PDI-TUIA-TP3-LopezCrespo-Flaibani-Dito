Desde la línea de comandos:

* git clone https://github.com/Facunditos/PDI-TUIA-TP3-LopezCrespo-Flaibani-Dito.git
* cd PDI-TUIA-TP3-LopezCrespo-Flaibani-Dito           # Cambia al directorio del proyecto
* python -m venv venv             # Crea el entorno virtual
* .\venv\Scripts\activate         # Activa el entorno virtual
* pip install -r requirements.txt # Instala las dependencias

**dados.py**

El programa consiste en analizar una tirada de dados para detectar el momento en que los dados quedan quietos, luego se calculan sus respectivos valores, para así determinar el resultado de la jugada. El script se encarga de analizar todos aquellos videos almacenados en la carpeta *videos* y de guardar los videos resultante del análsis en la carpeta *videos_procesados*. Por consola se imprime el resultado de la jugada. 