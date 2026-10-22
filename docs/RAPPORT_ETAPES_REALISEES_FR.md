# FlowMind - Rapport explicatif
Étapes réalisées

Projet SUMO et agent RL d’émissions | Mise à jour du 22 octobre 2026

Ce rapport explique le travail réalisé dans le projet local, depuis la collecte des métriques jusqu’au dernier entraînement. Il distingue les changements du programme, l’apprentissage du modèle et les résultats réellement mesurés. Les résultats du 21 octobre sont conservés comme historique ; les pages finales présentent la reprise du 22 octobre et la configuration actuellement active.

## 1. Le but de notre travail

Nous cherchons à piloter les feux de circulation pour diminuer les émissions des véhicules et la consommation de carburant. Nous comparons les décisions de l’agent à une référence avec des feux classiques.

| Métrique | Ce que nous mesurons | Unité du bilan |
| --- | --- | --- |
| CO₂ | Émissions de dioxyde de carbone | kg |
| NOx | Émissions d’oxydes d’azote | g |
| Carburant | Masse de carburant consommée | kg |
| PM / PMx | Particules calculées par le modèle d’émissions de SUMO | g |

PM signifie Particulate Matter, c’est-à-dire particules. Dans cette étude, PMx désigne la sortie d’émissions du modèle SUMO ; ce n’est pas une mesure réelle de la qualité de l’air.

## Le parcours suivi

1. Préparer un scénario et une référence avec des feux classiques.

2. Collecter et enregistrer les résultats de SUMO.

3. Construire un dashboard de simulation et d’historique.

4. Ajouter un agent à règles Python, puis un agent RL avec un modèle.

5. Entraîner, valider et tester le RL sur des seeds distinctes.

6. Corriger les protections contre les attentes excessives et renforcer le poids des PM.

7. Réentraîner et comparer le modèle retenu aux feux classiques.

# Préparer les essais et collecter les données

## 2. Définir une simulation comparable

Les derniers essais utilisent le scénario 3x3grid : un réseau de neuf carrefours, avec deux phases vertes par carrefour. Le réseau routier et les trajets des véhicules sont décrits dans scenarios/.

Chaque test dure 600 secondes, soit dix minutes simulées. Le fichier de routes fourni fait partir 300 véhicules pendant cette fenêtre. Le nombre de véhicules vient ici du fichier de routes : il n’a pas été choisi librement dans le formulaire pour ces essais.

Une seed est un nombre qui fixe les choix aléatoires de la simulation. Pour une comparaison, les méthodes utilisent la même carte, le même fichier de trafic, la même durée et la même seed. Changer la seed ne crée pas automatiquement une nouvelle demande de trafic.

## 3. Construire la référence

Dans le dashboard, « sans agent » correspond au contrôleur fixed_time : il fait tourner les phases avec 30 secondes de vert par phase, plus les transitions. Il ne s’adapte pas aux émissions et n’utilise pas de modèle appris. Le programme Python reste connecté à SUMO pour exécuter ce contrôle et mesurer les résultats.

## 4. Récupérer les métriques de SUMO

Le programme utilise TraCI, l’interface qui permet à Python de lire SUMO et de commander les feux. Il récupère des mesures chaque seconde, y compris pendant les transitions jaune et rouge.

| Fichier d’un essai | Rôle |
| --- | --- |
| timeseries.csv | Mesures au fil du temps : émissions, attente, files, vitesse, etc. |
| tripinfo.xml | Bilan officiel SUMO par trajet, avec les émissions des trajets non terminés à la fin. |
| metadata.json | Conditions de l’expérience : contrôleur, durée, seed, configuration et empreintes. |
| decisions.csv | Phases demandées et appliquées, avec une explication de la décision. |
| result.json | Indicateurs finaux calculés pour comparer les essais. |

Les taux instantanés, en mg/s, servent aux courbes. Les totaux finaux, en kg ou en g, proviennent du bilan officiel tripinfo. Ce ne sont pas les mêmes types de valeurs.

# Visualiser et prendre des décisions

## 5. Créer le dashboard

Le dashboard est une interface dans le navigateur. La page Simulation permet de choisir une méthode, de lancer ou arrêter un essai et de suivre les mesures. La page Historique affiche une courbe par métrique sur une fenêtre de temps choisie.

Les trois méthodes sont : feux classiques, agent à règles Python et agent avec modèle RL. Les courbes permettent de sélectionner une méthode ou de les comparer. SUMO conserve sa propre fenêtre lorsqu’on choisit le mode graphique.

Changer la méthode pendant une simulation permet d’observer son comportement. Pour mesurer une amélioration de façon comparable, nous utilisons cependant des essais séparés : une méthode hérite sinon de l’état de circulation laissé par la précédente.

## 6. Développer l’agent à règles Python

rules.py contient des règles écrites à la main. Son score utilise uniquement le CO₂, les NOx, le carburant et les PM. Il compare leur charge normalisée sur les voies des différentes phases et donne une priorité réactive à une direction autorisée.

Cet agent n’apprend pas. Les files, l’attente et la saturation restent utilisées pour les protections de circulation, mais elles ne sont pas des termes de son score environnemental. Un score fondé sur les émissions actuelles ne prédit pas à lui seul l’effet futur d’un changement de feu.

## 7. Ajouter un agent RL

| Élément | Fonctionnement du modèle actuel |
| --- | --- |
| Entrée | Émissions par phase, phase actuelle et durée du vert pour les neuf carrefours. |
| Sortie | Une proposition de phase par carrefour. Un seul modèle centralisé produit ces neuf choix. |
| Protection | SafetyGate vérifie les durées, les directions en attente et les sorties saturées. |
| Action appliquée | SUMO exécute la phase retenue avec les transitions prévues. |

Le modèle utilise PPO, un algorithme d’apprentissage par renforcement. Ce n’est pas un LLM. Dans le dashboard, il applique ce qu’il a appris ; il ne modifie pas ses paramètres automatiquement pendant son utilisation.

# Entraîner, valider et sauvegarder

## 8. Comprendre une boucle d’apprentissage

1. SUMO fournit l’état de la circulation et les émissions.

2. Le modèle propose les phases des feux ; les protections filtrent les décisions.

3. SUMO avance et le programme mesure les émissions produites.

4. Une récompense est calculée : moins d’émissions correspond à une valeur moins négative.

5. Pendant l’entraînement seulement, PPO utilise ces expériences pour mettre à jour le modèle.

Un épisode est une simulation d’apprentissage. Une étape PPO correspond à une décision du modèle, pas à une seconde SUMO : les transitions font varier le temps simulé entre deux décisions.

Un checkpoint est une sauvegarde du modèle à un moment de l’entraînement. Nous comparons plusieurs sauvegardes pour choisir celle à utiliser ensuite.

## 9. Séparer les données selon leur rôle

| Jeu | Utilisation |
| --- | --- |
| Entraînement | Produire des expériences et modifier le modèle. Seeds des premières campagnes : 0 à 5 ; reprise du 22 octobre : 6, 7, 8. |
| Validation | Choisir le checkpoint et le mode de décision. Première campagne : 1000, 1001 ; dernière reprise : 1002, 1003. |
| Test | Comparer sans apprentissage. Ancienne campagne : 10020 à 10022 ; dernier bilan : 10040, 10041, 10042. |

## 10. Les entraînements réalisés

| Version | Étapes exécutées au total | Checkpoint retenu | Objectif / évolution |
| --- | --- | --- | --- |
| Prototype | 128 | 128 | Vérifier le fonctionnement ; apprentissage très court. |
| Reprise V2 | 4 224 | 4 224 | 4 096 étapes ajoutées ; poids 1 / 1 / 1 / 1. |
| Reprise PM | 8 320 | 7 296 | 4 096 étapes ajoutées ; protections corrigées et PM ×2. |

Le dernier checkpoint n’est pas forcément le meilleur. Lors de la dernière session, celui de 7 296 étapes a été préféré à celui de 8 320 étapes sur la validation. 32 épisodes complets de dix minutes ont été terminés pendant cette session.

best.zip conserve le checkpoint retenu ; last.zip conserve le dernier état entraîné. manifest.json enregistre les poids, les protections, les seeds et les étapes. Les CSV du modèle conservent le suivi de l’apprentissage et de la validation.

# Corriger les blocages et renforcer les PM

## 11. Le problème constaté

L’ancien RL provoquait des téléportations : SUMO retirait temporairement des véhicules bloqués trop longtemps. Dans notre configuration, le seuil était de 60 secondes. Or la règle de priorité à une direction en attente pouvait intervenir après 90 secondes, trop tard pour ce seuil.

Sur les trois nouveaux essais de référence, l’ancien RL a produit 35 téléportations en moyenne. Ses émissions sont affectées par ces changements de présence des véhicules dans le réseau.

## 12. Les corrections réalisées

| Réglage du RL | Avant | Après |
| --- | --- | --- |
| Priorité à une direction en file au rouge | 90 s | 30 s |
| Vert maximal du RL | 60 s | 35 s |
| Vert minimal du RL | 10 s | 10 s |
| Seuil de téléportation SUMO | 60 s | 60 s, inchangé |
| Poids CO₂ / NOx / carburant / PM | 1 / 1 / 1 / 1 | 1 / 1 / 1 / 2 |

Une marge est gardée pour la prochaine décision, le jaune et la libération du carrefour. Les protections sont enregistrées avec le modèle et appliquées aussi à l’inférence. Le dashboard peut les renforcer, mais pas les relâcher au-delà des limites du modèle.

Le cycle des feux classiques reste à 30 secondes. Le seuil SUMO n’a pas été augmenté pour masquer les blocages. La priorité après 30 secondes concerne la mémoire de file d’une direction ; ce n’est pas une garantie sur le temps d’attente de chaque véhicule.

## 13. La nouvelle récompense

Les quatre quantités d’émissions sont normalisées pour rendre leurs échelles comparables. Le coût pondéré vaut : (CO₂ normalisé + NOx normalisé + carburant normalisé + 2 × PM normalisées) / 5. La récompense est l’opposé de ce coût.

Doubler le poids des PM les rend plus importantes dans l’objectif, sans garantir qu’elles baissent à chaque essai. Une moyenne pondérée accepte toujours des compromis entre les quatre métriques.

Pour retenir un checkpoint, nous minimisons d’abord les collisions, puis les téléportations, puis le coût environnemental. Le modèle protégé avant apprentissage a aussi été validé : cela distingue l’effet des protections de celui des nouvelles mises à jour.

# Historique : résultats du 21 octobre

Comparaison précédente : trois seeds (10020, 10021, 10022), 600 secondes et 300 véhicules partis par essai. Ces chiffres décrivent la version antérieure à la reprise du 22 octobre ; les résultats actuels figurent à la fin du rapport.

| Mesure | Classiques | RL corrigé | RL / classiques |
| --- | --- | --- | --- |
| CO₂ (kg) | 65.13 | 71.27 | +9.4 % |
| NOx (g) | 22.88 | 25.12 | +9.8 % |
| Carburant (kg) | 21.11 | 23.11 | +9.4 % |
| PM (g) | 4.86 | 4.78 | -1.8 % |
| Véhicules arrivés (véhicules) | 245.00 | 231.67 | -5.4 % |
| Téléportations (événements) | 0.00 | 0.00 | - |

![Comparaison des émissions](../runs/reports/pm_rl_20261021/emissions.png)

Le RL corrigé n’a enregistré aucune collision ni téléportation sur les trois tests. Les PM sont inférieures de 1,8 % à celles des feux classiques. En revanche, CO₂, NOx et carburant restent supérieurs : il ne bat pas encore la référence globalement.

Les PM restent environ 1,9 % au-dessus de celles de l’ancien RL. Il faut lire cette différence avec les téléportations de l’ancienne version. La correction des protections et le nouvel entraînement ont été combinés ; tout le changement ne vient pas du seul poids des PM.

# Retrouver le code et poursuivre l’étude

| Emplacement | Rôle |
| --- | --- |
| scenarios/ | Réseaux routiers et demandes de trafic SUMO. |
| config.yaml | Réglages des simulations, règles et protections spécifiques au RL. |
| flowmind/contracts.py | Structure des états et mesures échangés dans le programme. |
| flowmind/env.py | Connexion SUMO / TraCI, mesures et transitions des feux. |
| flowmind/agents/rules.py | Agent à règles avec score environnemental et contraintes. |
| flowmind/emission_rl.py | Entrées du modèle, protections et utilisation du PPO sauvegardé. |
| scripts/train_emissions.py | Entraînement, reprise, validation et choix des checkpoints. |
| flowmind/evaluation.py scripts/evaluate.py | Exécution et enregistrement des expériences comparables. |
| flowmind/kpis.py | Calcul des indicateurs et des variations entre méthodes. |
| scripts/dashboard.py dashboard/index.html | Serveur local et interface Simulation / Historique. |
| runs/models/emission_ppo/ | Modèle actif : checkpoint 7 296, mode probabiliste. Archive avant dernière reprise : emission_ppo_before_v4_20261021/. |
| runs/reports/rl_v4_20261022/ | Dernier bilan : rapport.md, comparaison.csv, details.json, activation.json et comparaison.png. |

## Ce qui est vérifié et ce qui reste à faire

Les 64 tests Python passent et la syntaxe JavaScript du dashboard a été vérifiée. Les neuf essais du dernier bilan ont produit zéro collision et zéro téléportation ; cela ne garantit pas l’absence de blocage sur tous les trafics.

Le modèle reste un prototype : un seul scénario et trois seeds de test ne suffisent pas à prouver sa généralisation. Il faut encore étudier des demandes et durées variées, améliorer les décisions sans dégrader une métrique et valider un débit de véhicules comparable aux feux classiques.

Pour ouvrir le dashboard, lancer scripts/dashboard.py avec Python, puis consulter http://127.0.0.1:8501/simulation ou /history. Cette adresse fonctionne lorsque le serveur est lancé. Les métriques sont des mesures de simulation, pas des mesures réelles de pollution.

Sources : code local sur manoubi ; manifest.json du modèle actif ; campagnes historiques dans runs/reports/ ; dernier bilan détaillé dans runs/reports/rl_v4_20261022/. scripts/validate_emission_sampling.py permet de valider le mode probabiliste avec les bilans officiels SUMO.

# Reprise du 22 octobre : méthode

## 14. Continuer l’entraînement sans imposer le dernier modèle

Nous avons repris le checkpoint de 7 296 étapes et exécuté 8 192 étapes supplémentaires, soit 64 nouveaux épisodes complets. Le compteur final atteint 15 488 étapes. Il repart du meilleur checkpoint chargé, et non du dernier état de la campagne précédente. Les poids restent 1 / 1 / 1 / 2 et les protections restent inchangées.

Les nouveaux checkpoints n’ont pas amélioré la validation. Ils ont donc été conservés comme traces d’expérience, sans remplacer les meilleurs paramètres du réseau. Entraîner plus longtemps ne garantit pas une amélioration.

## 15. Renforcer le choix du modèle

La validation compare désormais chaque candidat aux feux classiques sur les mêmes seeds, à partir des totaux officiels tripinfo.xml, trajets inachevés compris. Après les collisions et téléportations, elle minimise la plus forte dégradation relative parmi les quatre émissions et les déficits de véhicules arrivés ou partis. Le coût pondéré départage les égalités. La récompense d’apprentissage reste limitée aux quatre émissions.

## 16. Comparer deux façons d’utiliser PPO

Avant, le programme choisissait toujours l’action la plus probable. Le mode testé tire une action selon les probabilités apprises par le modèle : ce n’est pas un choix uniforme au hasard. Le tirage est reproductible avec une seed de politique égale à la seed SUMO + 100000 ; cet offset n’a pas été ajusté pour favoriser les résultats.

Le checkpoint et le mode ont été choisis uniquement sur la validation (1002, 1003), avant les tests finaux (10040, 10041, 10042). Le checkpoint précédent de 7 296 étapes avec tirage probabiliste a été retenu. Le rapprochement des résultats vers les feux classiques vient ici du mode de décision, pas d’une amélioration des poids par les 8 192 étapes ajoutées.

Le dashboard indique ce mode de décision et distingue les étapes exécutées du checkpoint retenu. Les résultats enregistrent également le mode et la seed de la politique. Une simulation d’évaluation ne réentraîne pas le modèle.

# Résultats actuels et limites

## 17. Comparaison finale avec les feux classiques

Trois essais de 600 secondes par méthode, sur 3x3grid, avec les mêmes conditions et les seeds 10040, 10041, 10042. Le bilan complet comprend neuf simulations : feux classiques, ancien RL déterministe et RL probabiliste retenu. Les valeurs ci-dessous sont les moyennes des trois essais.

| Métrique | Classiques | RL retenu | Variation |
| --- | --- | --- | --- |
| CO₂ (kg) | 64,8673 | 65,3981 | +0,82 % |
| NOx (g) | 22,7889 | 22,7685 | -0,09 % |
| Carburant (kg) | 21,0292 | 21,2012 | +0,82 % |
| PM (g) | 4,8722 | 4,8821 | +0,20 % |
| Véhicules arrivés | 246 | 244 | -0,81 % |
| Véhicules partis | 300 | 300 | 0 % |

Les pourcentages comparent le RL aux feux classiques : (RL - classiques) / classiques × 100. Pour les émissions, une valeur négative indique une baisse ; une valeur positive indique une hausse. Aucun des neuf essais n’a enregistré de collision ou de téléportation.

![Dernière comparaison](../runs/reports/rl_v4_20261022/comparaison.png)

L’objectif des quatre baisses n’est pas atteint. Les écarts sont faibles ; trois essais ne démontrent pas une supériorité générale. Le RL fait aussi arriver deux véhicules de moins en moyenne : ce débit doit rester visible dans l’interprétation.

# Configuration retenue et suite

## 18. Ce qui est actif et sauvegardé

La configuration retenue est active dans runs/models/emission_ppo/. Elle conserve les paramètres du checkpoint de 7 296 étapes, avec le mode probabiliste. Elle a été activée pour son meilleur résultat de validation et l’absence de collision et de téléportation dans les tests, et non parce que les quatre émissions seraient inférieures aux feux classiques.

| Emplacement dans runs/ | Contenu |
| --- | --- |
| models/emission_ppo_before_v4_20261021/ | Archive de la configuration précédente. |
| models/emission_ppo_v4_20261021/ | Nouvel entraînement, best.zip, last.zip et validations. |
| models/emission_ppo_v4_selected_20261022/ | Copie de la configuration retenue avant activation et traces du choix du mode. |
| reports/rl_v4_20261022/ | Résultats, identifiants des essais, comparaison CSV, graphique et décision d’activation. |

## 19. Interpréter et poursuivre

La configuration est expérimentale. Pour atteindre quatre réductions simultanées, il faut poursuivre les expériences sur les observations, les actions et la récompense, puis valider chaque proposition sans sélectionner le modèle sur les résultats de test. Davantage d’étapes seules n’ont pas suffi dans cette campagne.

Les prochaines comparaisons devront inclure plusieurs demandes de trafic, durées et seeds, ainsi que les arrivées et les trajets inachevés. Le poids PM renforcé accepte des compromis dans une récompense pondérée ; il n’impose pas mathématiquement la baisse de chaque polluant.

Les chiffres du 21 octobre et ceux du 22 octobre utilisent des seeds de test différentes. Pour mesurer directement l’évolution, le dernier graphique compare aussi l’ancien et le nouveau mode sur les mêmes trois seeds finales. Il faut éviter de soustraire des résultats provenant de campagnes différentes.
