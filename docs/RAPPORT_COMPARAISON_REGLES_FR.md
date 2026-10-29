# Comparaison des règles Python et des feux classiques — 22 octobre 2026

Les règles Python existantes ont été testées sans modification de leur score ou de leurs protections. Ce contrôleur ne charge aucun modèle RL.

## Conditions

Scénario 3x3grid, trafic fourni par routes14000.rou.xml, seeds 10090, 10091 et 10092 ; 600 secondes par essai ; deux répétitions pour chaque méthode et chaque seed, soit 12 simulations. Les départs, fichiers et paramètres du simulateur sont identiques dans chaque paire. L’ordre des méthodes est alterné. Les bilans incluent les véhicules qui n’ont pas terminé leur trajet.

Le score des règles utilise uniquement CO₂, NOx, carburant et PM, chacun avec un poids de 1. Il donne la priorité à la direction avec la charge d’émissions normalisée la plus élevée. Il ne prédit pas les émissions futures. Les contraintes actuelles sont conservées : vert minimal 10 s, vert maximal 60 s, priorité après 90 s d’attente, marge 0,05 et saturation de sortie 0,8. Le seuil de téléportation SUMO reste 60 s pour les deux méthodes.

## Résultats

| Métrique | Feux classiques, moyenne | Règles Python, moyenne | Variation moyenne appariée |
|---|---:|---:|---:|
| CO₂ | 65.645232 kg | 59.505487 kg | -9.351 % |
| NOx | 23.033553 g | 20.759223 g | -9.873 % |
| Carburant | 21.281353 kg | 19.290905 kg | -9.351 % |
| PM | 4.856603 g | 4.914688 g | +1.197 % |

Une variation négative signifie une baisse par rapport aux feux classiques. Le pourcentage est la moyenne des variations dans les six paires, et non le ratio des deux moyennes globales.

## Circulation et limites

| Méthode | Départs moyens | Arrivées moyennes | Événements de téléportation cumulés | Événements de collision cumulés |
|---|---:|---:|---:|---:|
| Feux classiques | 300.00 | 244.33 | 0 | 0 |
| Règles Python | 300.00 | 251.00 | 3 | 0 |

Une téléportation signifie que SUMO déplace automatiquement un véhicule bloqué. Elle peut modifier les temps de circulation et les émissions : une baisse observée avec des téléportations ne suffit pas à démontrer une amélioration du contrôleur. Les essais concernés restent dans le bilan ; ils ne sont pas supprimés pour améliorer les chiffres.

Les séries sont exactement identiques entre répétitions sur toutes les seeds : **False**. La reproductibilité générale de SUMO reste donc une limite à vérifier. Les amplitudes entre répétitions sont conservées dans comparaison.csv.

## Conclusion

Les baisses observées sur certaines émissions ne prouvent pas que le contrôleur améliore les quatre métriques. Les PM, les téléportations et la variabilité doivent être vérifiées avant de conclure à une amélioration fiable. Aucun paramètre de l’agent n’a été modifié pendant ces essais.

## Fichiers

Les résultats détaillés sont dans runs/reports/rules_comparison_20261022/ : trials.json, comparison.json, comparaison.csv et comparaison.png. Chaque run_id désigne aussi un dossier runs/<run_id>/ avec les séries temporelles, les décisions et les sorties officielles SUMO. Le dashboard peut afficher ces essais sous la méthode « Agent à règles Python ».

```powershell
python scripts/compare_emission_trials.py --controller rules --seeds 10090 10091 10092 --seconds 600 --repeats 2 --output runs/reports/nouvelle_comparaison_regles
```

Le dossier de sortie doit être nouveau. Les données runs/ sont ignorées par Git ; les scripts et ce rapport peuvent être partagés séparément des résultats et checkpoints.
