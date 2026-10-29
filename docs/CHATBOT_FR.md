# Agent d’explication des simulations

La page **Assistant** (`http://localhost:8501/chat`, onglet « Assistant ») interroge
un seul agent Gemini. On y choisit n’importe quelle simulation de l’historique
(la plus récente par défaut) ; l’assistant explique ses mesures, pendant son
exécution ou après sa fin. Une nouvelle simulation lancée entre-temps ne change
pas la simulation discutée. Il ne commande pas les feux et ne fait pas partie de l’orchestrateur des
agents de circulation.

## Configuration

Depuis la racine du projet :

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Renseigner une clé privée dans `.env`, à côté de `requirements.txt` :

```dotenv
GEMINI_API_KEY=remplacer_par_une_cle_privee
GEMINI_MODEL=gemini-3.5-flash-lite
```

Redémarrer ensuite le dashboard avec `python scripts/dashboard.py`. Les variables
déjà définies dans l’environnement priment sur `.env`. Ce fichier est exclu de
Git et n’est jamais servi par le dashboard. Ne pas réutiliser une clé publiée.
Le SDK et python-dotenv sont chargés sans empêcher la simulation de fonctionner
si la configuration Gemini est absente : le chatbot explique alors le problème.

## Données et consommation

- `flowmind/simulation_context.py` lit les données enregistrées côté serveur.
  Le navigateur fournit la question et l’identifiant affiché, jamais les faits.
- Les statistiques de la fenêtre récente portent sur les 120 dernières mesures
  complètes au maximum ; les séries CSV brutes ne sont pas envoyées à Gemini.
  La lecture parcourt le CSV, avec une mémoire bornée à 120 mesures.
- Le bilan final vient uniquement de `result.json`, calculé à partir des sorties
  officielles SUMO. Un essai arrêté ou échoué reste un bilan partiel.
- Chaque métrique inclut son unité. Les définitions distinguent attente courante
  et attente par trajet, émissions instantanées et totaux, vitesses récentes et
  vitesse finale pondérée par les véhicules-secondes.
- L’historique est limité à quatre messages, chacun tronqué à 800 caractères
  par l’interface. Il est effacé lorsqu’on choisit une autre simulation. La question
  courante accepte au plus 2 000 caractères.
- Le texte des instructions et des données est compté avec Gemini avant la
  génération. Une marge de 64 tokens couvre l’encapsulation ; le budget est de
  4 000 tokens d’entrée. L’historique est retiré en premier en cas de dépassement.
  Le nombre réel est également disponible dans les métadonnées de la réponse.
- La sortie est plafonnée à 600 tokens ; les consignes demandent 140 mots au plus.
  Une sortie signalée comme tronquée ou bloquée est traitée comme une erreur.
- Un cache local de 32 réponses maximum, valable 60 secondes, réutilise seulement
  les requêtes identiques (modèle, instructions, question, historique et faits).
  Toute nouvelle mesure modifie la clé du cache. Les tokens affichés pour une
  réponse réutilisée sont ceux de la génération initiale ; aucun nouvel appel
  de comptage ou génération n’est alors effectué.
- Une seule explication peut être générée à la fois ; la simulation continue
  indépendamment. Aucun appel Gemini n’est déclenché par le rafraîchissement des
  graphiques. Le délai réseau est borné. Une déconnexion pendant le comptage
  peut être réessayée une fois ; une génération n’est jamais relancée automatiquement.

## Limites d’interprétation

Aucune comparaison de référence n’est transmise dans cette première version.
L’agent doit donc refuser tout gain chiffré ou classement entre méthodes. Il
doit signaler les données manquantes et les changements de commande, et présenter
les causes possibles comme des hypothèses. Un résumé récent ne permet pas de
reconstituer une courbe entière ni une valeur à une seconde arbitraire.

Les consignes réduisent les erreurs d’interprétation, sans garantir l’exactitude
de toute réponse générée. Le texte doit être confronté aux métriques pour un
rapport scientifique. La limite de mots est une consigne ; la limite de tokens
est imposée par la requête API.

## Validation

```powershell
python -m pytest -q
python scripts/evaluate_explainer.py --run-id IDENTIFIANT
```

La seconde commande effectue jusqu’à dix-neuf appels réels de génération et leurs
comptages, avec la clé configurée. Elle utilise un essai enregistré et des cas
synthétiques clairement identifiés : données absentes, métrique manquante,
simulation arrêtée avec changements de commande, absence de référence et demande
d’inventer un résultat, ainsi que la distinction CO₂/carburant. Elle s’arrête à la première erreur API pour éviter de
multiplier les appels infructueux. Les questions, faits, réponses et consommations
sont conservés dans `runs/reports/chatbot_evaluation.json`, sans clé API.
Une réponse reçue n’équivaut pas à une validation factuelle : lire le rapport.
Pour choisir les cas à exécuter et consulter des questions à copier, voir
[Questions de test](QUESTIONS_TEST_CHATBOT_FR.md). Les options `--case` et
`--output` permettent une campagne ciblée sans écraser un rapport précédent.

Les tests automatisés utilisent des réponses simulées pour vérifier les unités,
les données incomplètes, le budget, le cache, les limites de concurrence, les
erreurs, la route HTTP et le refus des identifiants de simulation périmés.
