"""Thread-safe, validated commands applied at simulation decision boundaries."""
import math
import threading

RULE_LIMITS = {
    "co2_weight": (0, 20, 1), "nox_weight": (0, 20, 1),
    "fuel_weight": (0, 20, 1), "pm_weight": (0, 20, 1),
    "switch_margin": (0, 1, .05), "blocked_occupancy": (.1, 1, .8),
    "max_red": (10, 600, 90), "min_green": (3, 120, 10), "max_green": (3, 300, 60),
}


def validate_rules(values):
    if not isinstance(values, dict) or set(values) - set(RULE_LIMITS):
        raise ValueError("Paramètre de règle inconnu")
    result = {}
    for name, value in values.items():
        number = float(value)
        low, high, _ = RULE_LIMITS[name]
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{name} doit être compris entre {low} et {high}")
        result[name] = number
    return result


class RuntimeControl:
    def __init__(self, method="fixed_time", rules=None):
        self.lock = threading.Lock()
        self.state = {"method": "fixed_time", "rules": {k:v[2] for k,v in RULE_LIMITS.items()}, "revision": -1}
        self.update(method, rules or {})

    def snapshot(self):
        with self.lock:
            return self.state | {"rules": dict(self.state["rules"])}

    def update(self, method, rules):
        if method not in ("fixed_time", "rules", "emission_model"):
            raise ValueError("Choisissez sans agent, règles ou modèle émissions")
        values = validate_rules(rules)
        with self.lock:
            merged = self.state["rules"] | values
            if not any(merged[k + "_weight"] > 0 for k in ("co2", "nox", "fuel", "pm")):
                raise ValueError("Au moins un poids environnemental doit être positif")
            if merged["min_green"] > merged["max_green"]:
                raise ValueError("Le vert minimal doit être inférieur au vert maximal")
            if method != self.state["method"] or merged != self.state["rules"] or self.state["revision"] < 0:
                self.state = {"method": method, "rules": merged, "revision": self.state["revision"] + 1}
            return self.state | {"rules": dict(merged)}
