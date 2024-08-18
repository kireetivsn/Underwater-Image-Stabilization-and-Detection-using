
# Underwater image stabilization and detection using CNN

Underwater image stabilization is a process of removing colour cast from underwater images, mitigate colour distortion inherently present in the underwater images and adding back the lost colour to enhance the image . This is a step towards building an underwater surveillance system. The professional underwater cameras present right now are costly for any beginner to use. The images that these cameras take require additional processing in order for the general public to view and comprehend them. This project is an attempt towards making the process of enhancing the picture easier without the need of having a professional editor. This project is able to enhance underwater images effortlessly and the time taken to enhance them is done quickly. This project offers a web app GUI which makes it easier for general public or any recreational divers to use.This project is broadly divided in two modules , the first module stabilizes the picture using CLAHE  and unsharp masking and the second module detects the object present in the image using CNN  . The object can be classified into human, plants/corals, fish, sea bed.

for this project , install the following libraries of python

-streamlit, 
numpy,
matplotlib,
keras ,
open cv








## Installation and Run procedure

Install all the libraries by changing the below command

```bash
  pip install streamlit
```
to run the script , save the python script in a new folder , change the path to the folder via cd in cmd
```bash
   cd C:\Users\kiree\OneDrive\Desktop\engineering\underwater-image-stablization-main
```
make sure your python script is present in the path's folder
to run the script by streamlit via cmd
```bash
  streamlit run file_name.py
```

## Authors

- [@GaneshPVNS](https://www.linkedin.com/in/ganesh-p-n-v-s-23aa16254/)
- [@VarshaD](https://www.linkedin.com/in/donadula-varsha-62b38023b/)
