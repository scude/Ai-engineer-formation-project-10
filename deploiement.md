# Déploiement Azure Functions

Ce guide explique, pas à pas, comment déployer la fonction Azure `recommend` et configurer l’application Flask pour l’appeler.).

## 1) Prérequis locaux

Installez les outils nécessaires (Azure CLI + Azure Functions Core Tools) :

```bash
# Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Azure Functions Core Tools (v4)
# Doc officielle : https://learn.microsoft.com/azure/azure-functions/functions-run-local
sudo apt-get update
sudo apt-get install -y azure-functions-core-tools-4
```

Créez un environnement Python et installez les dépendances du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> **Astuce** : utilisez le login par device code :
> ```bash
> az login --use-device-code
> ```

## 2) Générer les artefacts ML

Les artefacts sont nécessaires au fonctionnement de la fonction Azure. Exécutez :

```bash
python -m src.train.build_artifacts
```

Les fichiers sont générés dans `artifacts/`.

## 3) Préparer le package Azure Functions

La fonction doit embarquer le code `src/` et les `artifacts/` pour fonctionner une fois déployée. Copiez-les dans le dossier de l’Azure Function :

```bash
rm -rf azure_function/RecommendFunction/src azure_function/RecommendFunction/artifacts
cp -r src azure_function/RecommendFunction/src
cp -r artifacts azure_function/RecommendFunction/artifacts
```

## 4) Créer les ressources Azure

Définissez quelques variables (adaptez la région si besoin) :

```bash
RESOURCE_GROUP="rg-reco"
LOCATION="westeurope"
STORAGE_ACCOUNT="reco$RANDOM$RANDOM"
FUNCTION_APP="reco-func-$RANDOM"
```

Créez le groupe de ressources et le compte de stockage :

```bash
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

az storage account create \
  --name "$STORAGE_ACCOUNT" \
  --location "$LOCATION" \
  --resource-group "$RESOURCE_GROUP" \
  --sku Standard_LRS
```

Créez l’Azure Function App (Linux, Python 3.11) :

```bash
az functionapp create \
  --name "$FUNCTION_APP" \
  --resource-group "$RESOURCE_GROUP" \
  --storage-account "$STORAGE_ACCOUNT" \
  --consumption-plan-location "$LOCATION" \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --os-type Linux
```

## 5) Déployer la fonction

Publiez la Function App depuis le dossier `azure_function/RecommendFunction` :

```bash
cd azure_function/RecommendFunction
func azure functionapp publish "$FUNCTION_APP"
cd -
```

Configurez les variables d’environnement nécessaires :

```bash
az functionapp config appsettings set \
  --name "$FUNCTION_APP" \
  --resource-group "$RESOURCE_GROUP" \
  --settings \
  ARTIFACTS_DIR="artifacts" \
  ALLOWED_ORIGIN="*"
```

> **Note CORS** : pour un environnement local, vous pouvez définir
> `ALLOWED_ORIGIN="http://127.0.0.1:5000"` (ou `http://localhost:5000` selon votre URL).
> Pour plusieurs origines, utilisez `ALLOWED_ORIGINS` avec une liste séparée par des virgules
> (ex. `ALLOWED_ORIGINS="http://127.0.0.1:5000,https://monapp.com"`).

## 6) Récupérer l’URL de la fonction

```bash
az functionapp function show \
  --name "$FUNCTION_APP" \
  --resource-group "$RESOURCE_GROUP" \
  --function-name recommend \
  --query "invokeUrlTemplate" \
  -o tsv
```

Vous obtiendrez une URL de type :

```
https://<function_app>.azurewebsites.net/api/recommend
```

## 7) Lancer l’application et appeler l’Azure Function

L’app Flask lit l’URL via la variable `AZURE_FUNCTION_URL`. Définissez-la avant de démarrer :

```bash
export AZURE_FUNCTION_URL="https://<function_app>.azurewebsites.net/api/recommend"
python app/app.py
```

Puis ouvrez l’app (ex. `http://localhost:5000`) et cliquez sur **Recommend** : l’appel est envoyé à la Function Azure.

---

## Dépannage rapide

- **La fonction ne trouve pas les artefacts** : vérifiez que `artifacts/` est bien présent sous `azure_function/RecommendFunction/` **avant** `func azure functionapp publish`, et que `ARTIFACTS_DIR=artifacts` est bien configuré.
- **Erreur CORS** : remplacez `ALLOWED_ORIGIN` par l’URL exacte de votre app frontend.
- **Temps de réponse long** : la première requête peut être plus lente (cold start).
