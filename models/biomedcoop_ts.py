"""
BiomedCoOp prompting strategy for MedTsLLM (decoder-only classification).

This is a *faithful* port of the BiomedCoOp mechanism
(https://github.com/HealthX-Lab/BiomedCoOp, Koleilat et al., CVPR 2025) into
MedTsLLM, replacing the earlier "inject descriptions as input text" approach
(which does not reproduce the method and did not help).

WHAT BIOMEDCOOP ACTUALLY DOES
-----------------------------
BiomedCoOp is a CLIP prompt-learning method. The LLM-generated per-class
descriptions are NOT fed to the model as input. They are encoded by a *frozen*
text encoder to produce per-class "teacher" text embeddings, which are then used
in three ways:

  1. As the **classifier weights**. In CoOp/CLIP the class text embeddings ARE
     the linear classifier: logits = scale * <sample_embedding, class_text_emb>.
     The descriptions therefore *define the decision boundary*.
  2. **KDSP** - Knowledge Distillation with Statistics-based Prompt Selection.
     Outlier descriptions are removed with a robust (median/MAD) z-score filter,
     the survivors are averaged into a per-class prototype, and the resulting
     zero-shot logits are distilled into the trainable head via KL divergence.
  3. **SCCM** - Semantic Consistency (an MSE pull of a *learnable* prompt's text
     embedding toward the mean teacher embedding). SCCM only applies when there
     is a learnable text prompt; it is optional here (off by default) because
     the minimal port trains the time-series side, not a text prompt.

WHY THE OLD PORT DIDN'T WORK
----------------------------
Concatenating descriptions into the decoder context produced a *constant*,
class-agnostic prefix in front of a *frozen* LLM whose classification head reads
only the time-series patch positions. A constant prefix carries no per-example,
label-discriminative gradient, so the head still had to do all the work from the
signal alone; random per-step sampling of descriptions merely added input noise.

THIS MODULE
-----------
`load_class_prompts(path, class_codes)`         -> list[list[str]]   (unchanged)
`statistics_based_prompt_selection(scores, tau)`-> boolean mask      (SPS)
`BiomedCoOpHead`                                -> nn.Module         (prototype
    classifier + KDSP/SCCM aux losses). The host model precomputes frozen
    per-class prototype embeddings from the descriptions and pools the LLM's
    time-series output into a sample embedding; the head does the rest.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "load_class_prompts",
    "statistics_based_prompt_selection",
    "BiomedCoOpHead",
    "BIOMEDCOOP_TS_TEMPLATES",
    "CODE_TO_NAME",
]


# ----------------------------------------------------------------------------
# Class descriptions (the BiomedCoOp prompt ensemble), adapted to the five
# PTB-XL diagnostic superclasses. Mirrors BiomedCoOp's BIOMEDCOOP_TEMPLATES
# (class -> list[str]).
# ----------------------------------------------------------------------------
CODE_TO_NAME = {
    "NORM": "Normal ECG",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy",
}

BIOMEDCOOP_TS_TEMPLATES = {
    "NORM": [
        "A normal ECG shows a regular sinus rhythm with a heart rate between 60 and 100 beats per minute.",
        "Each beat is preceded by an upright P wave in lead II, indicating normal sinus node activation.",
        "The PR interval is constant and falls within 120 to 200 milliseconds.",
        "The QRS complex is narrow, typically under 120 milliseconds, reflecting normal ventricular conduction.",
        "The ST segment is isoelectric, sitting level with the baseline with no elevation or depression.",
        "T waves are upright in most leads and concordant with the QRS complex.",
        "The QT interval is within normal limits when corrected for heart rate.",
        "R-wave progression across the precordial leads is smooth and orderly from V1 to V6.",
        "There are no pathological Q waves in a normal ECG.",
        "The rhythm is regular with even spacing between consecutive R waves.",
        "P waves are uniform in shape and precede every QRS complex one-to-one.",
        "The frontal-plane QRS axis falls within the normal range.",
        "No signs of chamber enlargement or hypertrophy are present.",
        "The baseline is stable without abnormal deflections between beats.",
        "A normal ECG has no ectopic beats or conduction blocks.",
        "Overall, the tracing shows organized, physiologic electrical activity with no abnormalities.",
    ],
    "MI": [
        "Myocardial infarction is characterized by ST-segment elevation in the leads facing the infarcted region.",
        "Pathological Q waves develop, reflecting established myocardial necrosis.",
        "In acute infarction, hyperacute peaked T waves may appear early before ST elevation.",
        "Reciprocal ST-segment depression is often seen in leads opposite the infarct.",
        "T-wave inversion evolves over time as the infarction progresses.",
        "Anterior infarction shows changes in the precordial leads V1 through V4.",
        "Inferior infarction produces changes in leads II, III, and aVF.",
        "Lateral infarction is reflected in leads I, aVL, V5, and V6.",
        "Loss of R-wave amplitude can accompany the development of Q waves.",
        "ST elevation in myocardial infarction is typically convex or dome-shaped.",
        "The location of ST elevation helps localize the coronary artery involved.",
        "Old or prior infarction is suggested by persistent Q waves without acute ST changes.",
        "The combination of Q waves, ST elevation, and T-wave inversion suggests an evolving infarction.",
        "Posterior infarction shows tall R waves and ST depression in V1 to V3 as reciprocal changes.",
        "A regional rather than diffuse pattern of abnormality is typical of infarction.",
        "Overall, myocardial infarction shows a regional pattern of ST-segment, Q-wave, and T-wave abnormalities.",
    ],
    "STTC": [
        "ST/T changes refer to abnormalities of the ST segment and T wave without diagnostic Q waves.",
        "ST-segment depression may indicate myocardial ischemia or strain.",
        "T-wave inversion can reflect ischemia, strain, or a non-specific repolarization change.",
        "Flattening of the T wave is a common non-specific ST/T abnormality.",
        "ST depression is often horizontal or downsloping in ischemia.",
        "Diffuse ST/T changes may be non-specific and not localize to a single territory.",
        "Repolarization abnormalities can be secondary to hypertrophy or conduction abnormalities.",
        "Symmetric, deep T-wave inversions may suggest active ischemia.",
        "ST/T changes can be dynamic and vary between recordings.",
        "Non-specific ST/T changes deviate from normal but do not meet criteria for infarction.",
        "ST-segment depression greater than one millimetre is generally considered significant.",
        "Subtle T-wave abnormalities may be the only sign of a repolarization disturbance.",
        "Drug effects and electrolyte imbalance can also produce ST/T changes.",
        "The ST segment may show subtle upsloping, downsloping, or horizontal depression.",
        "ST/T change describes repolarization abnormality without the injury current of infarction.",
        "Overall, ST/T change describes repolarization abnormalities of the ST segment and T wave.",
    ],
    "CD": [
        "Conduction disturbances involve delayed or blocked propagation of the cardiac impulse.",
        "Right bundle branch block produces a wide QRS with an rSR' pattern in V1.",
        "Left bundle branch block produces a wide QRS with broad, notched R waves in the lateral leads.",
        "First-degree AV block shows a prolonged PR interval beyond 200 milliseconds.",
        "Second-degree AV block shows intermittent failure of P waves to conduct to the ventricles.",
        "Third-degree (complete) AV block shows AV dissociation with independent P waves and QRS complexes.",
        "Bundle branch block widens the QRS complex beyond 120 milliseconds.",
        "Left anterior fascicular block causes left axis deviation with a characteristic QRS morphology.",
        "Left posterior fascicular block causes right axis deviation in the appropriate setting.",
        "Intraventricular conduction delay widens the QRS without meeting specific bundle branch criteria.",
        "In right bundle branch block, there is a wide, slurred S wave in leads I and V6.",
        "Conduction disturbances often produce secondary ST/T changes opposite to the QRS.",
        "Mobitz I (Wenckebach) block shows progressive PR prolongation before a dropped beat.",
        "Mobitz II block shows sudden non-conducted P waves without progressive PR change.",
        "The hallmark of conduction disturbance is an abnormal QRS width or AV relationship.",
        "Overall, conduction disturbances are recognized by widened QRS complexes or altered AV conduction.",
    ],
    "HYP": [
        "Hypertrophy on ECG reflects increased myocardial mass of the atria or ventricles.",
        "Left ventricular hypertrophy shows increased QRS voltage in the precordial leads.",
        "The Sokolow-Lyon criterion sums the S wave in V1 and the R wave in V5 or V6.",
        "Left ventricular hypertrophy is often accompanied by a strain pattern of ST depression and T inversion.",
        "Right ventricular hypertrophy shows a dominant R wave in V1 and right axis deviation.",
        "Left atrial enlargement produces a broad, notched P wave in lead II, known as P mitrale.",
        "Right atrial enlargement produces a tall, peaked P wave in lead II, known as P pulmonale.",
        "Increased R-wave amplitude in the lateral leads suggests left ventricular hypertrophy.",
        "Left ventricular hypertrophy may be associated with left axis deviation.",
        "The strain pattern shows downsloping ST depression with asymmetric T-wave inversion.",
        "Right ventricular hypertrophy may show a deep S wave in the lateral precordial leads.",
        "Voltage criteria combined with repolarization changes increase confidence in hypertrophy.",
        "Hypertrophy increases the amplitude and may widen the QRS complex.",
        "Biventricular hypertrophy shows features of both left and right ventricular enlargement.",
        "Chamber enlargement alters P-wave morphology and QRS voltage.",
        "Overall, hypertrophy is recognized by increased voltages and chamber-enlargement patterns.",
    ],
}


# ----------------------------------------------------------------------------
# Prompt loading (unchanged API)
# ----------------------------------------------------------------------------
def _normalize_templates(raw):
    if not isinstance(raw, dict):
        raise ValueError(
            "BiomedCoOp class-prompt file must be a JSON object mapping "
            "class code/name -> list of description strings."
        )
    out = {}
    for k, v in raw.items():
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, (list, tuple)) or len(v) == 0:
            raise ValueError(f"Class '{k}' must map to a non-empty list of strings.")
        out[k] = [str(s) for s in v]
    return out


def load_class_prompts(path, class_codes):
    """Return one list of description strings per class, ordered by class_codes.

    `path` may point to a JSON file (code/name -> list[str]) to override or
    extend the built-ins; pass "" / "none" / "default" to use the built-ins.
    """
    templates = dict(BIOMEDCOOP_TS_TEMPLATES)
    use_builtin = path is None or str(path).strip().lower() in (
        "", "none", "default", "builtin", "built-in",
    )
    if not use_builtin:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BiomedCoOp class-prompt file not found: {path!r}. Set "
                f'class_prompts_path to a valid JSON file, or to "" for built-ins.'
            )
        with open(path, "r", encoding="utf-8") as f:
            templates.update(_normalize_templates(json.load(f)))

    lookup = {str(k).strip().lower(): v for k, v in templates.items()}
    for code, name in CODE_TO_NAME.items():
        if code in templates and name.strip().lower() not in lookup:
            lookup[name.strip().lower()] = templates[code]

    descriptions = []
    for code in class_codes:
        key = str(code).strip().lower()
        if key not in lookup:
            name = CODE_TO_NAME.get(code)
            if name is not None and name.strip().lower() in lookup:
                key = name.strip().lower()
            else:
                raise KeyError(
                    f"No BiomedCoOp class prompts found for class {code!r}. "
                    f"Available keys: {sorted(templates.keys())}."
                )
        descriptions.append(list(lookup[key]))
    return descriptions


# ----------------------------------------------------------------------------
# Statistics-based Prompt Selection (SPS) -- faithful to BiomedCoOp
# ----------------------------------------------------------------------------
def statistics_based_prompt_selection(scores, tau):
    """Return a boolean mask over prompts, keeping non-outliers.

    Mirrors BiomedCoOp: a robust modified z-score using the median and the
    median absolute deviation (MAD), then a second standardisation, thresholded
    at `tau`. Falls back to keeping all prompts if MAD/std collapse.
    """
    scores = scores.detach().float()
    s_bar = torch.median(scores)
    d_bar = torch.median(torch.abs(scores - s_bar))
    if d_bar <= 0:
        return torch.ones_like(scores, dtype=torch.bool)
    z = (scores - s_bar) / d_bar
    z_std = torch.std(z)
    if z_std <= 0:
        return torch.ones_like(scores, dtype=torch.bool)
    mask = torch.abs((z - torch.mean(z)) / z_std) <= tau
    if mask.sum() == 0:                      # safety: never drop everything
        mask = torch.ones_like(scores, dtype=torch.bool)
    return mask


# ----------------------------------------------------------------------------
# BiomedCoOp prototype classification head
# ----------------------------------------------------------------------------
class BiomedCoOpHead(nn.Module):
    """Prototype (CLIP-style) classifier with KDSP / SCCM regularisation.

    The host model supplies:
      * `class_prototypes`: frozen per-class, per-prompt text embeddings of the
        descriptions, shape [n_cls, n_prompts, d_llm] (encoded once by the
        frozen LLM and pooled over tokens).
      * `ts_emb`: the pooled LLM output over the time-series patches, shape
        [bs, d_llm]  (the "sample" embedding, analogous to CLIP image features).

    Forward returns class logits. During training it also stores the auxiliary
    KDSP (+ optional SCCM) loss; the host model exposes it as `self.aux_loss`,
    which `tasks/classification.py` already adds to the cross-entropy loss.
    """

    def __init__(self, d_llm, n_cls, tau=2.0, kdsp_lambda=1.0, sccm_lambda=0.0,
                 logit_scale_init=4.6052, learnable_scale=True):
        super().__init__()
        self.n_cls = n_cls
        self.tau = tau
        self.kdsp_lambda = kdsp_lambda
        self.sccm_lambda = sccm_lambda

        # Projection of the time-series sample embedding into the (frozen) text
        # prototype space. Initialised near identity so training starts close to
        # a plain zero-shot prototype classifier.
        self.ts_proj = nn.Linear(d_llm, d_llm, bias=False)
        nn.init.eye_(self.ts_proj.weight)

        # Temperature, like CLIP's logit_scale (stored in log space).
        scale = torch.tensor(float(logit_scale_init))
        if learnable_scale:
            self.logit_scale = nn.Parameter(scale)
        else:
            self.register_buffer("logit_scale", scale)

        self.aux_loss = None  # set during training

    def forward(self, ts_emb, class_prototypes, labels=None):
        # ts_emb: [bs, d_llm] ; class_prototypes: [n_cls, n_prompts, d_llm]
        proto = class_prototypes.to(ts_emb.dtype)
        proto = proto / proto.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        proto_mean = proto.mean(dim=1)                                   # [n_cls, d_llm]
        proto_mean = proto_mean / proto_mean.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        scale = self.logit_scale.exp().clamp(max=100.0)

        # --- student head: learned projection vs mean prototypes ---
        feat = self.ts_proj(ts_emb)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        logits = scale * feat @ proto_mean.t()                          # [bs, n_cls]

        if labels is None or not self.training:
            self.aux_loss = None
            return logits

        # --- KDSP: distil SPS-selected zero-shot prototypes into the head ---
        with torch.no_grad():
            zs = ts_emb / ts_emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # zero-shot feature (no learned proj)
            # per-prompt alignment score s_p = mean_b max_c <zs, proto[:,p,:]>
            scores = []
            for p in range(proto.shape[1]):
                tl = scale * zs @ proto[:, p, :].t()                    # [bs, n_cls]
                scores.append(tl.max(dim=1).values.mean())
            scores = torch.stack(scores)                                # [n_prompts]
            mask = statistics_based_prompt_selection(scores, self.tau)
            selected = proto[:, mask, :].mean(dim=1)                    # [n_cls, d_llm]
            selected = selected / selected.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            zero_shot_logits = scale * zs @ selected.t()                # [bs, n_cls]

        loss_kdsp = F.kl_div(
            F.log_softmax(logits, dim=1),
            F.log_softmax(zero_shot_logits, dim=1),
            reduction="sum", log_target=True,
        ) / logits.numel()

        aux = self.kdsp_lambda * loss_kdsp

        # --- optional SCCM: pull mean prototypes & student head together ---
        if self.sccm_lambda > 0:
            aux = aux + self.sccm_lambda * F.mse_loss(feat.mean(0), proto_mean.mean(0))

        self.aux_loss = aux
        return logits
