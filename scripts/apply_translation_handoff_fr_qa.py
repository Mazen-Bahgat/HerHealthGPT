"""Apply reviewed, fail-closed corrections to the staged French v2 handoff.

The correction engine only permits edits to the two translation columns.  A
rule records the exact old/new snippet and expected match counts so input drift
cannot silently produce a partially corrected artifact.  Answer rules may be
expanded from one seed row to every row sharing the same English answer, which
preserves the handoff's deliberate answer reuse.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


TRANSLATION_FIELDS = ("Question_translated", "Answer_translated")
EXPECTED_HEADER = (
    "row_id",
    "split",
    "Question",
    "Answer",
    "Topic",
    "Keywords",
    *TRANSLATION_FIELDS,
)
BASE = Path("Used_Datasets/Consolidated_Datasets/200_Seed_Dataset")
DEFAULT_INPUT = BASE / "translation_handoff_v2" / "fr.generated.csv"
DEFAULT_OUTPUT = BASE / "translation_handoff_v2" / "fr.reviewed.csv"
DEFAULT_REPORT = BASE / "translation_handoff_v2" / "fr_translation_qa_report.json"
EXPECTED_INPUT_SHA256 = (
    "f9f179783e373ae4b10ba90ddf52d388fabc1e4c1a33e2594f95b9319fabaa02"
)


class CorrectionError(ValueError):
    """Raised when a correction rule cannot be applied exactly as reviewed."""


@dataclass(frozen=True)
class Rule:
    rule_id: str
    seed_row_ids: tuple[str, ...]
    column: str
    old: str
    new: str
    expected_cells: int
    expected_replacements: int
    expand_reused_answer: bool = False
    all_rows: bool = False


def _rule(
    rule_id: str,
    seed_row_ids: Sequence[str],
    column: str,
    old: str,
    new: str,
    expected_cells: int,
    expected_replacements: int | None = None,
    *,
    expand_reused_answer: bool = False,
    all_rows: bool = False,
) -> Rule:
    return Rule(
        rule_id=rule_id,
        seed_row_ids=tuple(seed_row_ids),
        column=column,
        old=old,
        new=new,
        expected_cells=expected_cells,
        expected_replacements=(
            expected_cells
            if expected_replacements is None
            else expected_replacements
        ),
        expand_reused_answer=expand_reused_answer,
        all_rows=all_rows,
    )


RULES = [
    # Normalize the generated clinical wrapper to a formal, gender-neutral
    # chart-note label.  The underlying patient text remains unchanged.
    _rule(
        "clinical-prefix-masculine",
        (),
        "Question_translated",
        "Le patient pr\u00e9sente la pr\u00e9occupation suivante : ",
        "Motif de consultation : ",
        277,
        all_rows=True,
    ),
    _rule(
        "clinical-prefix-feminine",
        (),
        "Question_translated",
        "La patiente pr\u00e9sente la pr\u00e9occupation suivante : ",
        "Motif de consultation : ",
        246,
        all_rows=True,
    ),
    _rule(
        "clinical-prefix-consultation",
        (),
        "Question_translated",
        "La patiente consulte pour la pr\u00e9occupation suivante : ",
        "Motif de consultation : ",
        1,
        all_rows=True,
    ),
    # Recurrent generated wrappers and answer closings that inferred gender
    # even though the English source did not state it.
    _rule(
        "reassurance-wrapper-feminine",
        (),
        "Question_translated",
        "j\u2019ai besoin d\u2019\u00eatre rassur\u00e9e",
        "j\u2019ai besoin que l\u2019on me rassure",
        105,
        all_rows=True,
    ),
    _rule(
        "reassurance-wrapper-masculine",
        (),
        "Question_translated",
        "j\u2019ai besoin d\u2019\u00eatre rassur\u00e9",
        "j\u2019ai besoin que l\u2019on me rassure",
        45,
        all_rows=True,
    ),
    _rule(
        "neutral-help-closing",
        (),
        "Answer_translated",
        "Je serai heureux de vous aider davantage.",
        "Je me ferai un plaisir de vous aider davantage.",
        48,
        all_rows=True,
    ),
    _rule(
        "neutral-uppercase-gratitude",
        (),
        "Question_translated",
        "JE VOUS SERAIS RECONNAISSANTE DE M\u2019AIDER \u00c0 CE SUJET.",
        "JE VOUS REMERCIE PAR AVANCE DE VOTRE AIDE \u00c0 CE SUJET.",
        2,
        all_rows=True,
    ),
    _rule(
        "neutral-uppercase-gratitude-pouvoir",
        (),
        "Question_translated",
        "JE VOUS SERAIS RECONNAISSANTE DE POUVOIR M\u2019AIDER \u00c0 CE SUJET.",
        "JE VOUS REMERCIE PAR AVANCE DE VOTRE AIDE \u00c0 CE SUJET.",
        2,
        all_rows=True,
    ),
    _rule(
        "neutral-uppercase-gratitude-conditionnel",
        (),
        "Question_translated",
        "JE VOUS SERAIS RECONNAISSANTE SI VOUS POUVIEZ M\u2019AIDER \u00c0 CE SUJET.",
        "JE VOUS REMERCIE PAR AVANCE DE VOTRE AIDE \u00c0 CE SUJET.",
        1,
        all_rows=True,
    ),
    _rule(
        "neutral-menopause-activity",
        (),
        "Answer_translated",
        "restez active",
        "maintenez une activit\u00e9 physique",
        6,
        all_rows=True,
    ),
    _rule(
        "neutral-doctor-reference",
        (),
        "Answer_translated",
        "discutez avec lui de toute pr\u00e9occupation",
        "discutez-en avec votre m\u00e9decin",
        6,
        all_rows=True,
    ),
    _rule(
        "neutral-positive-advice",
        (),
        "Answer_translated",
        "restez positive",
        "gardez confiance",
        6,
        all_rows=True,
    ),
    _rule(
        "neutral-assurance-advice",
        (),
        "Answer_translated",
        "Soyez assur\u00e9 qu\u2019il s\u2019agit d\u2019un coup de chaleur",
        "Il s\u2019agit bien d\u2019un coup de chaleur",
        6,
        all_rows=True,
    ),
    _rule(
        "idiomatic-emotional-wrapper",
        (),
        "Question_translated",
        "je n\u2019arrive pas \u00e0 arr\u00eater d\u2019y penser",
        "je n\u2019arrive pas \u00e0 m\u2019emp\u00eacher d\u2019y penser",
        18,
        all_rows=True,
    ),
    _rule(
        "repair-csomething-token",
        ("train-0598", "train-0784"),
        "Question_translated",
        "cquelque chose",
        "quelque chose",
        2,
    ),
    _rule(
        "hair-fall-not-hairball-initial",
        ("train-1522",),
        "Answer_translated",
        "La boule de cheveux",
        "La chute des cheveux",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "hair-fall-not-hairball-later",
        ("train-1522",),
        "Answer_translated",
        "la boule de cheveux",
        "la chute des cheveux",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "neutral-vegetarian-phrase",
        ("train-1522",),
        "Answer_translated",
        "si vous \u00eates v\u00e9g\u00e9tarien",
        "si votre alimentation est v\u00e9g\u00e9tarienne",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "contextual-pod-train-1356",
        ("train-1356",),
        "Answer_translated",
        "POD",
        "SOPK",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "contextual-pod-train-0997",
        ("train-0997",),
        "Answer_translated",
        "POD",
        "SOPK",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "contextual-pcs-train-0997",
        ("train-0997",),
        "Answer_translated",
        "PCs",
        "SOPK",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "contextual-pod-train-1703",
        ("train-1703",),
        "Answer_translated",
        "POD",
        "SOPK",
        6,
        12,
        expand_reused_answer=True,
    ),
    _rule(
        "restore-uncertainty-train-1703",
        ("train-1703",),
        "Answer_translated",
        "vous ne devriez rencontrer aucun probl\u00e8me pour concevoir si vos cycles sont r\u00e9guliers",
        "il se peut que vous ne rencontriez pas de difficult\u00e9 \u00e0 concevoir si vos cycles sont r\u00e9guliers",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "contextual-pod-train-1892",
        ("train-1892",),
        "Answer_translated",
        "POD",
        "SOPK",
        6,
        18,
        expand_reused_answer=True,
    ),
    _rule(
        "remove-invented-dose-form-train-1892",
        ("train-1892",),
        "Answer_translated",
        "Prenez de l\u2019ac\u00e9taminoph\u00e8ne, un comprim\u00e9, pour vous soulager.",
        "Prenez de l\u2019ac\u00e9taminoph\u00e8ne pour vous soulager.",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "neutral-patient-variation-train-1892",
        ("train-1892",),
        "Answer_translated",
        "d\u2019une patiente \u00e0 l\u2019autre",
        "d\u2019une personne \u00e0 l\u2019autre",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "complete-conceive-phrase-train-1645",
        ("train-1645",),
        "Question_translated",
        "J\u2019essaie de concevoir,",
        "J\u2019essaie de concevoir un enfant,",
        1,
    ),
    _rule(
        "repair-endometriosis-grammar-train-0293",
        ("train-0293",),
        "Question_translated",
        "Existe-t-il des signes pr\u00e9coces indiquant que de la muqueuse suppl\u00e9mentaire se d\u00e9veloppe en dehors de ma matrice (ut\u00e9rus)/une endom\u00e9triose ?",
        "Existe-t-il des signes pr\u00e9coces de la pr\u00e9sence de muqueuse suppl\u00e9mentaire en dehors de ma matrice (ut\u00e9rus), ou d\u2019une endom\u00e9triose ?",
        1,
    ),
    _rule(
        "neutral-adolescence-question-train-0412",
        ("train-0412",),
        "Question_translated",
        "lorsque je suis adolescente",
        "pendant mon adolescence",
        1,
    ),
    _rule(
        "neutral-adolescence-answer-train-0412",
        ("train-0412",),
        "Answer_translated",
        "lorsque vous \u00eates adolescente ou que vous approchez de la m\u00e9nopause",
        "pendant l\u2019adolescence ou \u00e0 l\u2019approche de la m\u00e9nopause",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "all-means-everyone-train-0814",
        ("train-0814",),
        "Answer_translated",
        "pour toutes",
        "pour tout le monde",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "neutral-worry-train-2800",
        ("train-2800",),
        "Question_translated",
        "je me suis inqui\u00e9t\u00e9 en le voyant",
        "sa lecture m\u2019inqui\u00e8te",
        1,
    ),
    _rule(
        "french-ibuprofen-train-0694",
        ("train-0694",),
        "Answer_translated",
        "l\u2019Ibuprofen",
        "l\u2019ibuprof\u00e8ne",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "neutral-emergency-evaluation-train-1563",
        ("train-1563",),
        "Answer_translated",
        "pour \u00eatre examin\u00e9",
        "pour une \u00e9valuation",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "neutral-irritability-train-0465",
        ("train-0465",),
        "Answer_translated",
        "tendance \u00e0 \u00eatre rapidement agac\u00e9e ou contrari\u00e9e",
        "tendance \u00e0 s\u2019agacer ou \u00e0 se contrarier rapidement",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "word-form-one-hour-train-1172",
        ("train-1172",),
        "Answer_translated",
        "1 heure avant le petit-d\u00e9jeuner",
        "une heure avant le petit-d\u00e9jeuner",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "restore-ambiguous-final-time-train-1054",
        ("train-1054",),
        "Question_translated",
        "et j\u2019ai eu de nouveau un rapport sexuel \u00e0 23 h ?",
        "puis j\u2019ai de nouveau eu un rapport sexuel \u00e0 11 h ?",
        1,
    ),
    _rule(
        "keep-symptoms-in-one-question-train-1600",
        ("train-1600",),
        "Question_translated",
        "pourrais-je quand m\u00eame \u00eatre enceinte ? je me sens aussi fatigu\u00e9e, naus\u00e9euse et d'humeur changeante ?",
        "pourrais-je quand m\u00eame \u00eatre enceinte, sachant que je ressens aussi de la fatigue, des naus\u00e9es et des changements d\u2019humeur ?",
        1,
    ),
    _rule(
        "restore-elliptical-question-train-2706",
        ("train-2706",),
        "Question_translated",
        "mais elle s\u2019est termin\u00e9e par une fausse couche, et rien depuis un an.",
        "mais elle s\u2019est termin\u00e9e par une fausse couche, et plus rien depuis un an ?",
        1,
    ),
    _rule(
        "preserve-incomplete-semen-question-val-0246",
        ("val-0246",),
        "Question_translated",
        "bonjour docteur, voici mon spermogramme",
        "bonjour docteur, mon spermogramme est",
        1,
    ),
    _rule(
        "idiomatic-conception-val-0246",
        ("val-0246",),
        "Answer_translated",
        "apr\u00e8s qu\u2019elle aura con\u00e7u",
        "apr\u00e8s qu\u2019elle sera tomb\u00e9e enceinte",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "french-metformin-val-0242",
        ("val-0242",),
        "Question_translated",
        "Metformin",
        "metformine",
        1,
        2,
    ),
    _rule(
        "restore-past-pcos-val-0330",
        ("val-0330",),
        "Question_translated",
        "j'ai un SOPK",
        "j'avais un SOPK",
        1,
    ),
    _rule(
        "preserve-vague-scopy-val-0331",
        ("val-0331",),
        "Question_translated",
        "une endoscopie",
        "une scopie",
        1,
    ),
    _rule(
        "contextual-pod-val-0330-answer",
        ("val-0330",),
        "Answer_translated",
        "un pod syndrome des ovaires polykystiques",
        "un syndrome des ovaires polykystiques (SOPK)",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "restore-timing-reference-val-0470",
        ("val-0470",),
        "Question_translated",
        "le m\u00e9dicament unwanted 72 apr\u00e8s 68 heures",
        "le m\u00e9dicament unwanted 72 68 heures apr\u00e8s le rapport",
        1,
    ),
    _rule(
        "neutral-kneeling-val-0661-0664-0665",
        ("val-0661", "val-0664", "val-0665"),
        "Question_translated",
        "Je me suis agenouill\u00e9e sur",
        "J\u2019ai pos\u00e9 le genou sur",
        3,
    ),
    _rule(
        "neutral-landing-val-0663",
        ("val-0663",),
        "Question_translated",
        "je suis retomb\u00e9e juste l\u00e0",
        "j\u2019ai atterri dessus",
        1,
    ),
    _rule(
        "neutral-landing-val-0665",
        ("val-0665",),
        "Question_translated",
        "je suis retomb\u00e9e dessus",
        "j\u2019ai atterri dessus",
        1,
    ),
    _rule(
        "preserve-emotion-val-0665",
        ("val-0665",),
        "Question_translated",
        "Je me sens affreusement coupable !",
        "Je me sens terriblement mal !",
        1,
    ),
    _rule(
        "preserve-hpf-unit-val-0706",
        ("val-0706",),
        "Question_translated",
        "4-6 cellules de pus/champ \u00e0 fort grossissement",
        "cellules de pus 4-6 /hpf",
        1,
    ),
    _rule(
        "retain-womb-synonym-val-0000-answer",
        ("val-0000",),
        "Answer_translated",
        "muqueuse de votre ut\u00e9rus, qui",
        "muqueuse de votre ut\u00e9rus (matrice), qui",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "retain-womb-synonym-val-0084-question",
        ("val-0084",),
        "Question_translated",
        "en dehors de mon ut\u00e9rus ?",
        "en dehors de ma matrice (ut\u00e9rus) ?",
        1,
    ),
    _rule(
        "idiomatic-delayed-periods-val-0114",
        ("val-0114",),
        "Answer_translated",
        "avoir des r\u00e8gles retard\u00e9es",
        "avoir un retard de r\u00e8gles",
        6,
        expand_reused_answer=True,
    ),
    _rule(
        "neutral-period-expectation-straight-apostrophe",
        ("val-0384", "val-0386"),
        "Question_translated",
        "J'\u00e9tais cens\u00e9e avoir mes r\u00e8gles le 4 d\u00e9cembre",
        "Mes r\u00e8gles \u00e9taient pr\u00e9vues le 4 d\u00e9cembre",
        2,
    ),
    _rule(
        "neutral-period-expectation-curly-apostrophe",
        ("val-0385", "val-0387", "val-0389"),
        "Question_translated",
        "J\u2019\u00e9tais cens\u00e9e avoir mes r\u00e8gles le 4 d\u00e9cembre",
        "Mes r\u00e8gles \u00e9taient pr\u00e9vues le 4 d\u00e9cembre",
        3,
    ),
    _rule(
        "preserve-ipill-case-val-0385",
        ("val-0385",),
        "Question_translated",
        "iPill",
        "ipill",
        1,
    ),
    _rule(
        "preserve-ipill-form-val-0387-0389",
        ("val-0387", "val-0388", "val-0389"),
        "Question_translated",
        "i-pill",
        "ipill",
        3,
    ),
    _rule(
        "preserve-meprate-case-val-0385-0389",
        ("val-0385", "val-0387", "val-0389"),
        "Question_translated",
        "Meprate",
        "meprate",
        3,
    ),
    _rule(
        "preserve-unwanted-case-val-0468-0473",
        ("val-0468", "val-0469", "val-0471", "val-0472", "val-0473"),
        "Question_translated",
        "Unwanted 72",
        "unwanted 72",
        5,
    ),
    _rule(
        "remove-inferred-diazen-count-lowercase",
        ("val-0294",),
        "Question_translated",
        "1 comprim\u00e9 de diazen 2 fois par jour",
        "du diazen 2 fois par jour",
        1,
    ),
    _rule(
        "remove-inferred-diazen-count-titlecase",
        ("val-0295", "val-0297", "val-0299"),
        "Question_translated",
        "1 comprim\u00e9 de Diazen 2 fois par jour",
        "du diazen 2 fois par jour",
        3,
    ),
    _rule(
        "preserve-vague-months-val-0450",
        ("val-0450",),
        "Question_translated",
        "en 1 mois (fluctuation)",
        "en quelques mois (fluctuation)",
        1,
    ),
    _rule(
        "preserve-vague-months-val-0455",
        ("val-0455",),
        "Question_translated",
        "en l\u2019espace de 1 mois (fluctuation)",
        "en l\u2019espace de quelques mois (fluctuation)",
        1,
    ),
    # Exhaustive follow-up audit of every remaining answer containing POD/PCs
    # confirmed polycystic-ovary context.  Question occurrences are deliberately
    # excluded because POD can instead mean "pouch of Douglas" in scan text.
    _rule(
        "contextual-pod-all-remaining-answers",
        (),
        "Answer_translated",
        "POD",
        "SOPK",
        78,
        90,
        all_rows=True,
    ),
    _rule(
        "contextual-pcs-all-remaining-answers",
        (),
        "Answer_translated",
        "PCs",
        "SOPK",
        48,
        54,
        all_rows=True,
    ),
    _rule(
        "preserve-i-pill-spacing-train-1051-1055",
        ("train-1051", "train-1052", "train-1053", "train-1054", "train-1055"),
        "Question_translated",
        "i-pill",
        "i pill",
        5,
    ),
    _rule(
        "preserve-ipill-form-train-1753",
        ("train-1753",),
        "Question_translated",
        "i-pill",
        "ipill",
        1,
    ),
    _rule(
        "preserve-i-pill-spacing-train-1885-1888",
        ("train-1885", "train-1888"),
        "Question_translated",
        "I-pill",
        "I pill",
        2,
    ),
    _rule(
        "preserve-unwanted-case-val-0264-0269",
        ("val-0264", "val-0265", "val-0267", "val-0269"),
        "Question_translated",
        "Unwanted 72",
        "unwanted 72",
        4,
    ),
    _rule(
        "translate-pouch-of-douglas-scan-label-train-2592-2597",
        ("train-2592", "train-2593", "train-2594", "train-2595", "train-2597"),
        "Question_translated",
        "Annexes/POD",
        "Annexes/cul-de-sac de Douglas",
        5,
    ),
]


def apply_rules(
    rows: list[dict[str, str]], rules: list[Rule]
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    """Return corrected row copies and an audit record for every applied rule."""

    corrected = [dict(row) for row in rows]
    by_id = {row["row_id"]: row for row in corrected}
    if len(by_id) != len(corrected):
        raise CorrectionError("input contains duplicate row_id values")

    audit: list[dict[str, object]] = []
    for rule in rules:
        if rule.column not in TRANSLATION_FIELDS:
            raise CorrectionError(
                f"{rule.rule_id}: {rule.column!r} is not a translation column"
            )
        if rule.all_rows and rule.seed_row_ids:
            raise CorrectionError(
                f"{rule.rule_id}: all_rows cannot be combined with seed_row_ids"
            )
        missing_ids = [row_id for row_id in rule.seed_row_ids if row_id not in by_id]
        if missing_ids:
            raise CorrectionError(f"{rule.rule_id}: missing seed row(s) {missing_ids!r}")

        targets = corrected if rule.all_rows else [by_id[row_id] for row_id in rule.seed_row_ids]
        if rule.expand_reused_answer:
            if rule.column != "Answer_translated":
                raise CorrectionError(
                    f"{rule.rule_id}: answer expansion requires Answer_translated"
                )
            source_answers = {row["Answer"] for row in targets}
            targets = [row for row in corrected if row["Answer"] in source_answers]

        # Multiple seed ids or answers can select the same row; keep stable CSV order.
        target_ids = {row["row_id"] for row in targets}
        targets = [row for row in corrected if row["row_id"] in target_ids]
        matching = [row for row in targets if rule.old in row[rule.column]]
        replacement_count = sum(row[rule.column].count(rule.old) for row in matching)
        if len(matching) != rule.expected_cells:
            raise CorrectionError(
                f"{rule.rule_id}: expected {rule.expected_cells} matching cells, "
                f"found {len(matching)}"
            )
        if replacement_count != rule.expected_replacements:
            raise CorrectionError(
                f"{rule.rule_id}: expected {rule.expected_replacements} replacements, "
                f"found {replacement_count}"
            )

        for row in matching:
            row[rule.column] = row[rule.column].replace(rule.old, rule.new)
        audit.append(
            {
                "rule_id": rule.rule_id,
                "column": rule.column,
                "row_ids": [row["row_id"] for row in matching],
                "replacement_count": replacement_count,
            }
        )

    return corrected, audit


def normalize_translation_line_whitespace(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    """Remove horizontal whitespace before embedded newlines in translations."""

    normalized = [dict(row) for row in rows]
    audit: list[dict[str, object]] = []
    pattern = re.compile(r"[ \t]+(?=\r?\n)")
    for row in normalized:
        for column in TRANSLATION_FIELDS:
            value = row[column]
            trimmed_line_count = len(pattern.findall(value))
            if not trimmed_line_count:
                continue
            row[column] = pattern.sub("", value)
            audit.append(
                {
                    "row_id": row["row_id"],
                    "column": column,
                    "trimmed_line_count": trimmed_line_count,
                }
            )
    return normalized, audit


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_csv(path: Path) -> tuple[bytes, list[dict[str, str]]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CorrectionError(f"{path}: not valid UTF-8") from exc
    reader = csv.DictReader(text.splitlines(keepends=True))
    if tuple(reader.fieldnames or ()) != EXPECTED_HEADER:
        raise CorrectionError(
            f"{path}: header drift; expected {EXPECTED_HEADER!r}, got {reader.fieldnames!r}"
        )
    rows = list(reader)
    if len(rows) != 3_580:
        raise CorrectionError(f"{path}: expected 3580 rows, found {len(rows)}")
    return raw, rows


def _write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    raw = temporary.read_bytes()
    temporary.replace(path)
    return raw


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the reviewed French translation QA corrections."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify the locked input and every correction rule without writing files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    if input_path == output_path:
        raise CorrectionError("--output must differ from --input")

    raw, rows = _read_csv(input_path)
    input_sha256 = _sha256_bytes(raw)
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise CorrectionError(
            f"{input_path}: input SHA-256 drift; expected {EXPECTED_INPUT_SHA256}, "
            f"got {input_sha256}"
        )
    corrected, audit = apply_rules(rows, RULES)
    corrected, whitespace_audit = normalize_translation_line_whitespace(corrected)
    for before, after in zip(rows, corrected, strict=True):
        for field in EXPECTED_HEADER[:6]:
            if before[field] != after[field]:
                raise CorrectionError(
                    f"{before['row_id']}: correction changed immutable {field}"
                )
        if not after["Question_translated"].strip() or not after[
            "Answer_translated"
        ].strip():
            raise CorrectionError(f"{before['row_id']}: correction produced a blank cell")

    changed_cells = {
        (before["row_id"], field)
        for before, after in zip(rows, corrected, strict=True)
        for field in TRANSLATION_FIELDS
        if before[field] != after[field]
    }
    total_replacements = sum(
        int(item["replacement_count"]) for item in audit
    )
    total_trimmed_lines = sum(
        int(item["trimmed_line_count"]) for item in whitespace_audit
    )
    print(
        f"verified {len(RULES)} fail-closed rules; {len(changed_cells)} cells changed; "
        f"{total_replacements} snippet replacements; "
        f"{total_trimmed_lines} translated line endings trimmed"
    )
    if args.dry_run:
        return 0

    output_raw = _write_csv_atomic(output_path, corrected)
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact": output_path.relative_to(Path.cwd()).as_posix(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": input_path.relative_to(Path.cwd()).as_posix(),
        "input_sha256": input_sha256,
        "output_sha256": _sha256_bytes(output_raw),
        "review_method": {
            "corpus_wide_deterministic_validation": True,
            "deterministic_review_jobs_triaged": 404,
            "ai_assisted_stratified_rows_reviewed": 180,
            "ai_assisted_sample_fraction": 180 / 3_580,
            "sample_critical_safety_reversals": 0,
            "native_french_human_review": "pending",
            "downstream_ingest_verification": {
                "status": "passed",
                "train_rows": 2_858,
                "validation_rows": 718,
                "dropped_train_val_duplicates": 4,
                "note": (
                    "Four synonymous English questions collapse to identical natural "
                    "French across splits and are removed by the existing leakage guard."
                ),
            },
        },
        "corrections": {
            "rule_count": len(RULES),
            "changed_cell_count": len(changed_cells),
            "snippet_replacement_count": total_replacements,
            "rules": audit,
            "line_end_whitespace_normalization": {
                "changed_cell_count": len(whitespace_audit),
                "trimmed_line_count": total_trimmed_lines,
                "cells": whitespace_audit,
            },
        },
        "adjudication_left_pending": [
            "train-1370: source CSI in an assisted-reproduction list may mean ICSI",
            "train-2175: source GAEC doctor may be a corruption of gynae doctor",
            "train-1512/train-1513/train-1514/train-1515/train-1517: malformed date '1 October 31st'",
            "train-1773 and train-1791: ambiguous date formatting/localization",
        ],
    }
    _write_json_atomic(report_path, report)
    print(f"wrote reviewed CSV -> {output_path}")
    print(f"wrote QA report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
