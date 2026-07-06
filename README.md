# farptest

This repo contains all of the code necessary to engage in the weekly FARP point defense challenge. At beginning of each week,
an updated version of problem will be uploaded. Submissions should be made Friday of the same week. Look below for instructions
to get started.

## Quickstart
```
git clone https://github.com/GMU-ASRC/farptest
cd farptest
uv venv
source .venv/bin/activate  # activate env (see below)
uv pip install -r requirements.txt
python eval_genome.py -- [0.2, 0.2, 0.2, -0.2]
```

Depending on your OS and shell, you may need to use a different activation command. Make sure you're in the `farptest/` directory:

Shell | OS | Activation Command
-|-|-
CMD.exe 	| (Windows)	 | `.\.venv\Scripts\activate`
PowerShell 	| (Windows)	 | `.\.venv\Scripts\activate.ps1`
NuShell 	| (Windows)	 | `overlay use .\.venv\Scripts\activate.nu`
bash/zsh	| (Unix/Mac) | `source .venv/bin/activate`
Fish 		| (Unix/Mac) | `source .venv/bin/activate.fish`
NuShell | (Unix/Mac) | `overlay use .venv/bin/activate.nu`

If you would like to plot your results, run the following before opening `graph_results.ipynb`
```
uv pip install -r jupyter.txt
```
