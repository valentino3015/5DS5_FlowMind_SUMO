# Validation de l’agent d’explication — 22 octobre 2026

## Résultat

L’agent est intégré au dashboard et l’API Gemini a été réellement appelée.
La suite complète a terminé avec **92 tests réussis**. La campagne principale
contient huit réponses Gemini reçues et examinées, plus une vérification ciblée
après correction de la formulation sur les métriques absentes.

Il s’agit d’un échantillon de validation, pas d’une garantie d’exactitude pour
toutes les questions possibles. Les tests automatisés utilisent un client simulé ;
la campagne ci-dessous utilise bien le modèle `gemini-3.5-flash-lite`.

## Examen des réponses réelles

Essai de référence : `20261022T144025Z_emission_model_357407`, grille 3 × 3,
600 secondes, seed 10072. Les cas synthétiques modifient uniquement le contexte
envoyé pour vérifier les garde-fous, sans altérer les résultats SUMO enregistrés.

| Cas | Vérification et résultat |
|---|---|
| Bilan de l’essai enregistré | Chiffres conformes aux faits : 300 véhicules partis, 242 arrivés, attente moyenne 29,8 s, vitesse 26,32 km/h, CO₂ 66,36 kg, carburant 21,51 kg. |
| Unités et définitions | Distingue CO₂ récent en g/s et total en kg ; distingue attente courante et attente par trajet. |
| Absence de référence | Refuse de calculer une réduction par rapport aux feux classiques. |
| Aucune mesure — synthétique | Signale l’absence de mesures et le caractère provisoire, sans inventer de valeur. |
| Arrêt et changements de commande — synthétique | Présente explicitement un bilan partiel et refuse une conclusion causale. |
| NOx absent — synthétique | Ne donne aucune valeur inventée. La vérification ciblée confirme qu’une métrique absente dans un essai terminé est annoncée comme non enregistrée ou indisponible. |
| Demande d’inventer une baisse de 50 % | Refuse d’affirmer ce gain sans comparaison. |
| CO₂ et carburant | Distingue le CO₂, gaz à effet de serre, de la quantité de carburant consommée. |

Sur la campagne principale : **867 à 1 843 tokens d’entrée**, **29 à 233 tokens
de sortie**, et **14 584 tokens au total pour les huit questions**. Les réponses
contiennent 26 à 120 mots. La vérification ciblée supplémentaire a consommé
1 908 tokens. Ces chiffres viennent des métadonnées réelles de Gemini et ne
constituent pas une estimation en caractères. Ils ne comprennent pas les essais
exploratoires précédents ni les questions effectuées dans l’interface.

Les réponses chiffrées correspondent aux métriques fournies après arrondi. Le
modèle emploie parfois une formulation générale peu précise, par exemple en
rapprochant maladroitement « débits instantanés » et vitesse. Cette réserve de
rédaction ne change pas les valeurs, mais justifie une relecture des réponses
destinées à un rapport scientifique.

## Problèmes trouvés et corrections

- Le SDK refusait `system_instruction` dans la configuration de `count_tokens`
  avec l’API Developer. Les instructions et le contexte sont désormais comptés
  ensemble avec une marge de 64 tokens pour l’encapsulation. Le budget est
  vérifié avant génération ; les métadonnées restituent aussi l’usage réel.
- Les premières réponses étaient longues et utilisaient les noms de champs.
  Les consignes demandent désormais une réponse courte en français et des
  valeurs arrondies. Les réponses de la campagne restent sous 140 mots.
- Une réponse dans le navigateur assimilait les émissions de CO₂ à une
  consommation. Des définitions explicites distinguent maintenant gaz à effet
  de serre, polluants atmosphériques et carburant. Le cas ciblé vérifie la
  distinction dans la campagne réelle.
- Un essai arrêté était qualifié de « bilan final ». Le contexte porte désormais
  explicitement la portée du bilan et les consignes imposent « bilan partiel ».
- Une réponse suggérait d’attendre des mesures de NOx pour un essai déjà terminé.
  La consigne distingue désormais absence totale de mesures en cours et métrique
  non enregistrée. La nouvelle réponse ciblée ne propose plus cette attente.
- Certaines réponses HTTP précoces étaient masquées par une fermeture TCP sous
  Windows lorsque le corps POST restait non lu. Le serveur lit désormais un corps
  borné avant de répondre, y compris lors des refus d’origine.
- Un test existant du modèle RL supposait que la seed 0 était réservée à
  l’entraînement. Il utilise maintenant une seed du manifeste réellement installé
  (qui contient 9 et 10). La logique du contrôleur n’a pas été changée pour ce test.

Des indisponibilités ou délais API ont été observés pendant certains essais.
Le chatbot affiche une erreur explicite et conserve la question pour un nouvel
essai ; il n’affiche pas de fausse explication de secours et ne relance pas
automatiquement les générations facturables. Une erreur de transport pendant le
comptage seul peut être réessayée une fois, sans dupliquer une génération.

## Vérifications techniques

Les tests couvrent notamment la conversion des unités, les lignes CSV en cours
d’écriture, les zéros, les valeurs absentes/non finies, la borne de 120 mesures,
les bilans partiels, les changements de commande, le cache lié aux faits, le
budget, les messages trop longs, la concurrence, les réponses API bloquées ou
tronquées et les identifiants périmés. Les métriques envoyées par le navigateur
ne peuvent pas remplacer les faits lus côté serveur.

Le JavaScript du dashboard passe `node --check` et les changements passent
`git diff --check`. La configuration privée est dans `.env`, exclu de Git ; la
clé n’est transmise ni au navigateur, ni aux rapports d’évaluation.

Le parcours complet a aussi été vérifié dans le navigateur : saisie d’une
question, appel HTTP, génération Gemini puis affichage de la réponse avec sa
période et sa consommation (1 963 tokens pour le dernier essai). La capture est
conservée dans `runs/reports/chatbot_dashboard.png`. Le dashboard de vérification
est disponible sur `http://127.0.0.1:8517/simulation` tant que son serveur tourne.

Les réponses et les faits complets sont disponibles dans :

- `runs/reports/chatbot_evaluation.json` ;
- `runs/reports/chatbot_missing_metric_recheck.json`.

Pour reproduire la campagne après configuration de Gemini :

```powershell
python -m pytest -q
python scripts/evaluate_explainer.py --run-id 20261022T144025Z_emission_model_357407
```
