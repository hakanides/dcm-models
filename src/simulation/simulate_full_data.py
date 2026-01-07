import argparse
import json
import numpy as np
import pandas as pd


def softmax(u):
    u = u - np.max(u)
    e = np.exp(u)
    return e / e.sum()

def likert_from_latent(rng, loading, lv, thresholds):
    y_star = loading * lv + rng.standard_normal()
    t1, t2, t3, t4 = thresholds
    if y_star <= t1: return 1
    if y_star <= t2: return 2
    if y_star <= t3: return 3
    if y_star <= t4: return 4
    return 5

def enforce_sign(value, sign):
    if sign is None:
        return value
    if sign == "positive":
        return abs(value)
    if sign == "negative":
        return -abs(value)
    raise ValueError(f"Unknown enforce_sign='{sign}' (use positive/negative/null)")

def draw_categorical(rng, spec):
    vals = np.array(spec["values"])
    probs = np.array(spec["probs"], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(vals, p=probs)

def build_demographics(rng, demog_cfg):
    demo = {}
    for k, spec in demog_cfg.items():
        if spec["type"] == "categorical":
            demo[k] = int(draw_categorical(rng, spec))
        else:
            raise ValueError(f"Unknown demog type for {k}: {spec['type']}")
    return demo

def build_latents(rng, latent_cfg, demo):
    lat = {}
    for lv_name, spec in latent_cfg["structural"].items():
        mu = float(spec.get("intercept", 0.0))
        betas = spec.get("betas", {})
        for x, b in betas.items():
            mu += float(b) * float(demo.get(x, 0.0))
        sigma = float(spec.get("sigma", 1.0))
        lat[lv_name] = mu + sigma * rng.standard_normal()
    return lat

def factor_to_lv_name(factor_str):
    fac = str(factor_str).strip().lower()
    if fac == "blind":
        return "pat_blind"
    if fac == "constructive":
        return "pat_constructive"
    if fac in ("daily", "dl", "dailylife"):
        return "sec_dl"
    if fac in ("faith", "fp", "faithandprayer", "faith_prayer"):
        return "sec_fp"
    raise ValueError(f"Unknown factor '{factor_str}' in items_config.csv")


def term_value(term, row, alt_name, fee_scale):
    """Return attribute value x for a given alternative.
    Supports fee10k, fee100k, and fee_scaled naming variants."""
    if alt_name == "paid1":
        fee = float(row.fee1); dur = float(row.dur1)
    elif alt_name == "paid2":
        fee = float(row.fee2); dur = float(row.dur2)
    elif alt_name == "standard":
        fee = float(row.fee3); dur = float(row.dur3)
    else:
        raise ValueError(f"Unknown alt_name={alt_name}")

    if term == "const":
        return 1.0

    # Fee scalings (compat)
    if term in ("fee100k", "fee1e5", "fee_scaled"):
        return fee / float(fee_scale)
    if term == "fee10k":
        return fee / 10000.0
    if term in ("fee",):
        # fallback: interpret as scaled by fee_scale
        return fee / float(fee_scale)

    if term == "dur":
        return dur

    raise ValueError(f"Unknown term '{term}'")


def compute_utility(cfg_choice, row, alt_name, demo, lat):
    """Utility with robust sign handling:
    U += beta_i(term) * x(term)
    where beta_i = base_coef + sum_j coef_j * z_j (with optional centering/scaling),
    and enforce_sign is applied to the *total* beta_i (not only to interaction parts).
    """
    fee_scale = float(cfg_choice.get("fee_scale", 100000.0))
    U = 0.0

    # Alternative-specific base terms (e.g., ASCs) as before
    for bt in cfg_choice.get("base_terms", []):
        if alt_name in bt["apply_to"]:
            coef = float(bt["coef"])
            U += coef * term_value(bt["term"], row, alt_name, fee_scale)

    # Attribute terms: build total beta first, then apply to x
    for at in cfg_choice.get("attribute_terms", []):
        if alt_name not in at["apply_to"]:
            continue

        term = at["term"]
        x = term_value(term, row, alt_name, fee_scale)

        # total marginal coefficient for this term (person-specific)
        beta = float(at.get("base_coef", 0.0))

        for it in at.get("interactions", []):
            cov = it["with"]
            coef = enforce_sign(float(it.get("coef", 0.0)), it.get("enforce_sign", None))

            if cov in demo:
                z = float(demo[cov])
            elif cov in lat:
                z = float(lat[cov])
            else:
                raise ValueError(f"Interaction covariate '{cov}' not found in demo or latents.")

            # Optional centering/scaling (recommended for indices like income_idx)
            if "center" in it and it["center"] is not None:
                z -= float(it["center"])
            if "scale" in it and it["scale"] not in (None, 0, 0.0):
                z /= float(it["scale"])

            beta += coef * z

        # Enforce sign on the TOTAL beta (this is the key fix)
        desired = at.get("enforce_sign", None)
        if desired is None:
            # sensible defaults if not explicitly given in config
            if term in ("fee100k", "fee1e5", "fee_scaled", "fee10k", "fee"):
                desired = "negative"
            elif term == "dur":
                desired = "negative"
        beta = enforce_sign(beta, desired)

        U += beta * x

    return U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/model_config.json")
    ap.add_argument("--out", default="synthetic_hcm_panel.csv")
    ap.add_argument("--keep_latent", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    N = int(cfg["population"]["N"])
    T = int(cfg["population"]["T"])
    seed = int(cfg["population"]["seed"])
    rng = np.random.default_rng(seed)

    design_path = cfg["design"]["path"]
    base = pd.read_csv(design_path)

    design_cols = ["scenario_id", "dur1", "fee1", "dur2", "fee2", "dur3", "fee3"]
    missing = [c for c in design_cols if c not in base.columns]
    if missing:
        raise ValueError(f"Design missing columns: {missing}")

    design = base[design_cols].drop_duplicates("scenario_id").reset_index(drop=True)
    if len(design) < T:
        raise ValueError(f"Need at least T={T} unique scenarios; only {len(design)} in design.")

    items_path = cfg["measurement"]["items_path"]
    items = pd.read_csv(items_path)
    required = {"item_name","scale","factor","reverse","loading"}
    if not required.issubset(set(items.columns)):
        raise ValueError(f"items_config.csv must have columns: {sorted(required)}")

    thresholds = tuple(cfg["measurement"]["thresholds"])
    if len(thresholds) != 4:
        raise ValueError("measurement.thresholds must be a list of 4 cutpoints")

    demog_cfg = cfg["demographics"]
    latent_cfg = cfg["latent"]
    choice_cfg = cfg["choice_model"]
    alts = choice_cfg["alts"]

    out_rows = []

    for i in range(1, N + 1):
        ID_STR = f"SYN_{i:06d}"

        demo = build_demographics(rng, demog_cfg)
        lat = build_latents(rng, latent_cfg, demo)

        likert_answers = {}
        for r in items.itertuples(index=False):
            lv_name = factor_to_lv_name(r.factor)
            y = likert_from_latent(
                rng=rng,
                loading=float(r.loading),
                lv=float(lat[lv_name]),
                thresholds=thresholds,
            )
            if int(r.reverse) == 1:
                y = 6 - y
            likert_answers[str(r.item_name)] = int(y)

        sampled = design.sample(n=T, replace=False, random_state=int(rng.integers(1, 1_000_000_000)))

        for t, row in enumerate(sampled.itertuples(index=False), start=1):
            U = []
            for alt_code in ["1","2","3"]:
                alt_name = alts[alt_code]
                U.append(compute_utility(choice_cfg, row, alt_name, demo, lat))

            p = softmax(np.array(U))
            choice = int(rng.choice([1,2,3], p=p))

            rec = {
                "ID_STR": ID_STR,
                "ID": i,
                "task": t,
                "scenario_id": int(row.scenario_id),
                "dur1": float(row.dur1), "fee1": float(row.fee1),
                "dur2": float(row.dur2), "fee2": float(row.fee2),
                "dur3": float(row.dur3), "fee3": float(row.fee3),
                "CHOICE": choice,
                "bad_task": 0,
            }

            rec.update({k: demo[k] for k in demo})
            rec.update(likert_answers)

            if args.keep_latent:
                for k, v in lat.items():
                    rec[f"LV_{k}_true"] = float(v)

            out_rows.append(rec)

    synth = pd.DataFrame(out_rows)
    synth.to_csv(args.out, index=False)

    print("=== DONE ===")
    print("Wrote:", args.out)
    print("Rows:", len(synth), "Respondents:", synth["ID"].nunique())
    print("Choice shares:", synth["CHOICE"].value_counts(normalize=True).sort_index().to_dict())
    print("Items:", len(items), "Reverse items:", int(items["reverse"].sum()))

if __name__ == "__main__":
    main()
