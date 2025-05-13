### Install requirements

```
pip install -r requirements.txt
```

### Download and extract data

```
gdown 1LwXqOnw0iCUK28cn6DCa2hdz3a_MgYBZ -O data/UBFC.tar
tar -xvf data/UBFC.tar -C data
```

### Run pipeline

```
dvc repro
```

### Viewing metrics and plots

```
dvc metrics show
dvc plots show --open
```

### Run linter

```
ruff check src
isort src
```
