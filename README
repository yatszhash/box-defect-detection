
## Box deficit detection models

## report

You can see the model report [here](./analyze/report.ipynb)

## get trained model and result

Download zip file from 

https://drive.google.com/open?id=1sB-zyM0wzE-yyrVSwSUmqNEFScUnjYvQ

and unzip it into the project root.
This step are required to infer with the trained models.

## DataSet

I'm sorry that the dataset is not allowed to publish.  

## How to train?

### prerequieste
- anadonda3

### install dependencies
In the project root, 

```
conda env create -f=environment.yml
```

### execute train script
In the project root, 

```
export PYTHONPATH=.
python scripts/train.py $MODEL_NAME
```

chose $MODEL_NAME from 
- "resnet50_pretrained"
- "resnet50_pretrained_cosine"
- "resnet50_pretrained_cosine_pruned"

"resnet50_pretrained_cosine_pruned" model have to be trained after training "resnet50_pretrained_cosine".

These train results are saved under [output dir](output)"

##  How to Infer with trained model?

Use  *Inferrer Class* in [inference.py](inference.py).

Please see [inference_sample.py](inference_sample.py) to see how to use it.  
