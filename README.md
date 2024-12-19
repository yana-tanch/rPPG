# rPPG

## Установка

- Клонирование репозитория

``` shell
git clone https://github.com/yana-tanch/rPPG.git
```

- Создание окружения

``` shell
conda env create -n rppg --file conda_rppg_env.yaml
conda activate rppg
```

## Данные
Использовался датасет UBFC-rPPG

Структура данных:

```
rPPG-Toolbox-main/data/UBFC-rPPG/
         |   |-- subject1/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |   |-- subject2/
         |       |-- vid.avi
         |       |-- ground_truth.txt

```

## Запуск

``` shell
cd rPPG-Toolbox-main/

# запуск метода POS с вычислением метрики MAE
python main.py --config_file .configs/infer_configs/UBFC-rPPG_UNSUPERVISED.yaml
```