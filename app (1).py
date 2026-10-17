"""Application Gradio — prédiction de souscription à un dépôt à terme (Bank Marketing).

Le modèle `modele_bank.joblib` est un Pipeline scikit-learn complet
(encodage + normalisation + classifieur) produit par le notebook.
"""
import os
import tempfile

import gradio as gr
import joblib
import pandas as pd

DOSSIER = os.path.dirname(os.path.abspath(__file__))
MODELE = joblib.load(os.path.join(DOSSIER, "modele_bank.joblib"))

# Colonnes attendues et modalités des variables catégorielles, lues directement dans le pipeline
COLONNES = list(MODELE.feature_names_in_)
_prep = MODELE.named_steps["prep"]
CATEGORIES = dict(zip(_prep.transformers_[0][2], _prep.named_transformers_["cat"].categories_))

VALEURS_PAR_DEFAUT = {"age": 40, "duration": 240, "campaign": 2, "pdays": 999, "previous": 0}


def predire_un(*valeurs):
    """Prédiction pour un seul client (formulaire)."""
    if any(v is None or v == "" for v in valeurs):
        raise gr.Error("Veuillez renseigner tous les champs.")
    x = pd.DataFrame([dict(zip(COLONNES, valeurs))])
    p = float(MODELE.predict_proba(x)[0, 1])
    return {"Dépôt": p, "Pas de dépôt": 1 - p}


def predire_csv(fichier):
    """Prédiction pour plusieurs clients (fichier CSV)."""
    if fichier is None:
        raise gr.Error("Importez d'abord un fichier CSV.")
    chemin = getattr(fichier, "name", fichier)
    try:
        donnees = pd.read_csv(chemin)
    except Exception as e:
        raise gr.Error(f"Fichier CSV illisible : {e}")

    manquantes = [c for c in COLONNES if c not in donnees.columns]
    if manquantes:
        raise gr.Error("Colonnes manquantes dans le CSV : " + ", ".join(manquantes))

    proba = MODELE.predict_proba(donnees[COLONNES])[:, 1]
    resultat = donnees.copy()
    resultat["prediction"] = ["Dépôt" if p >= 0.5 else "Pas de dépôt" for p in proba]
    resultat["probabilite_depot"] = proba.round(4)

    sortie = os.path.join(tempfile.mkdtemp(), "predictions_avec_resultats.csv")
    resultat.to_csv(sortie, index=False)
    return sortie, resultat.head(20)


def _champ(col):
    if col in CATEGORIES:
        choix = [str(c) for c in CATEGORIES[col]]
        return gr.Dropdown(choices=choix, value=choix[0], label=col)
    return gr.Number(value=VALEURS_PAR_DEFAUT.get(col, 0), label=col, precision=0)


interface_simple = gr.Interface(
    fn=predire_un,
    inputs=[_champ(c) for c in COLONNES],
    outputs=gr.Label(num_top_classes=2, label="Probabilité de souscription"),
    title="Prédire la souscription bancaire — un client",
    description=(
        "Ce modèle de Machine Learning estime la probabilité qu'un client souscrive "
        "un dépôt à terme, à partir de son profil et des informations sur la campagne."
    ),
    flagging_mode="never",
)

interface_multiple = gr.Interface(
    fn=predire_csv,
    inputs=gr.File(label="Importer un fichier CSV", file_types=[".csv"]),
    outputs=[
        gr.File(label="Télécharger le CSV avec les prédictions"),
        gr.Dataframe(label="Aperçu (20 premières lignes)"),
    ],
    title="Prédire la souscription bancaire — plusieurs clients",
    description="Le CSV doit contenir les colonnes : " + ", ".join(COLONNES) + ".",
    flagging_mode="never",
)

demo = gr.TabbedInterface(
    [interface_simple, interface_multiple],
    ["Prédiction simple", "Prédiction multiple"],
    title="Bank Marketing — Dépôt à terme",
)

if __name__ == "__main__":
    demo.launch()
