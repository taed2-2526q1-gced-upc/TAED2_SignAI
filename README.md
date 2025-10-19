# SignAI

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Signal identification based on Neural Networks

---

## Project Overview
**SignAI** is a machine learning system for automatic **traffic sign detection and classification** using the YOLOv8 model.
The project integrates a **FastAPI backend** for model inference and a **Streamlit frontend** that provides a simple graphical interface for user interaction.

---

## Features
- Real-time **traffic sign detection** using YOLOv8.
- **FastAPI** backend for efficient and scalable model deployment.
- **Streamlit** web interface for intuitive user interaction.
- Modular design with **unit and integration tests** for reliability.
- Fully reproducible environment using **uv** or `requirements.txt`.

---

## Project Organization
```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         scr and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── scr   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes scr a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```


---

## Running the Application

To run the complete **SignAI** system, both the **API** and the **frontend** must be launched from the project’s root directory.

### 1 Start the API (FastAPI)

Run the API server using:

```bash
python -m uvicorn scr.app.api:app --reload --port 8080
```

Once running, the API will be available at:
`http://localhost:8080`

---

### 2 Start the Frontend (Streamlit)

Launch the web interface with:

```bash
streamlit run ./scr/app/frontend.py
```

⚠️ **Important:** The API must be running **before** launching the Streamlit interface, since the frontend sends requests to the `/predict` endpoint for inference.

## 👥 Authors

This project was developed as part of the course **Temes Avançats d’Enginyeria de Dades II (TAED2)**,
within the **BSc in Data Science and Engineering** at the **Universitat Politècnica de Catalunya (UPC)**
during the **2025–2026 academic year**.

**Authors:**

- **Laia Villagrasa** — [laia.villagrasa@estudiantat.upc.edu](mailto:laia.villagrasa@estudiantat.upc.edu)
- **Maria Poveda** — [maria.poveda@estudiantat.upc.edu](mailto:maria.poveda@estudiantat.upc.edu)




--------

