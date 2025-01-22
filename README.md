# Examen DVC et Dagshub
Dans ce dépôt vous trouverez l'architecture proposé pour mettre en place la solution de l'examen. 

```bash       
├── examen_dvc          
│   ├── data
│   │   ├── normalized_data      
│   │   ├── processed_data      
│   │   └── raw _data      
│   ├── metrics       
│   ├── models      
│   │   ├── data      
│   │   └── models        
│   ├── src
│   │   ├── data
│   │   |    ├── split_data.py      
│   │   |    └── normalized_data.py     
│   │   └── models
│   │        ├── evaluate_model.py
│   │        ├── grid_search.py    
│   │        └── train_model.py       
│   └── README.md.py
│   └── dvc.lock
│   └── dvc.yaml
│   └── READ.md #Contains Name, Email and Dagshub Repository information 
│   └── requirements.txt      
```
N'hésitez pas à rajouter les dossiers ou les fichiers qui vous semblent pertinents.

Vous devez dans un premier temps *Fork* le repo et puis le cloner pour travailler dessus. Le rendu de cet examen sera le lien vers votre dépôt sur DagsHub. Faites attention à bien mettre https://dagshub.com/licence.pedago en tant que colaborateur avec des droits de lecture seulement pour que ce soit corrigé.

Vous pouvez télécharger les données à travers le lien suivant : https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv.
