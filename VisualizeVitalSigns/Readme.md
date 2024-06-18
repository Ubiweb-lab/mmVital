# Technical Stack and Constraints
As estimating vital signs from radar signals using deep learning models is one of the research focuses, Python is chosen as the backend programming language to facilitate easier integration with deep learning models in the future and enhance software development efficiency. The specific technology stack and constraints are as follows:

- Back-end 
  - Primary language: Python
  - Framework used: Django
  - Description: Django is a high-level web framework for the Python programming language that enables the rapid development of secure and maintainable websites. In this project, we utilize Django as the backend server to process the data and return the data of vital signs to the client.
- Front-end
  - Primary language: Javascript, HTML5, CSS
  - Framework used: jQuery, Echart, TailwindCSS
  - Description: The TailwindCSS framework is utilized to control the layout, such as colors, fonts, and overall appearance of the visualization page. The framework jQuery and Echart are utilized for the data request and response processing between the front-end and back-end, as well as the waveform presentation of heart and respiratory rates.
- Data Storage Format
  - CSV File
  - Description: CSV files are used to save the data of the vital signs of each participant.

# How to run the application?
To run the application, you must first set up the Python environment and install the Django backend framework. 

1. go to your project path
```sh
cd /path/your project path
```
2. run the project
```sh
python manage.py runserver
```
3. access the visualization page, type the address in your browser: http://127.0.0.1:8000/
   

**Note: if you would like to access the visualization page through your mobile phone, you need to add an IP address format when you start the application.**
```sh
python manage.py runserver 0.0.0.0:8000
```
then, Find your LAN IP address on the computer where you are running the application.
<img width="457" alt="image" src="https://github.com/Ubiweb-lab/mmVital/assets/39370733/6092fe36-1e74-4641-b510-7aa55ca5941b">

Lastly, access the visualization page, type the address in your mobile phone browser: http://172.29.28.17:8000/
