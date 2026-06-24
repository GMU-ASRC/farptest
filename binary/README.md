# farptest

This repo contains all of the code necessary to engage in the weekly FARP point defense challenge. At beginning of each week,
an updated version of problem will be uploaded. Submissions should be made Friday of the same week. Look below for instructions
to get started.

## Quickstart
```
uv venv
source .venv/bin/activate
git clone https://github.com/GMU-ASRC/farptest
cd farptest
uv pip install -r requirements.txt
python eval_genome.py -- [0.2, 0.2, 0.2, -0.2]
```

If you would like to plot your results, run the following before opening `graph_results.ipynb`
```
uv pip install -r jupyter.txt
```
