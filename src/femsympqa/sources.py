from pathlib import Path

from pydantic import BaseModel

RAW_DIR = Path("data/raw")

CONDITIONS: dict[str, str] = {
    "pcos": "PCOS",
    "heavy-periods": "Heavy periods (menorrhagia)",
    "infertility": "Infertility (female)",
    "endometriosis": "Endometriosis",
}


class Source(BaseModel):
    slug: str
    condition: str
    url: str


SOURCES: list[Source] = [
    Source(slug="nhs-pcos", condition=CONDITIONS["pcos"],
           url="https://www.nhs.uk/conditions/polyendocrine-metabolic-ovarian-syndrome-pmos/"),
    Source(slug="cdc-pcos", condition=CONDITIONS["pcos"],
           url="https://www.cdc.gov/diabetes/basics/pcos.html"),
    Source(slug="nhs-heavy-periods", condition=CONDITIONS["heavy-periods"],
           url="https://www.nhs.uk/conditions/heavy-periods/"),
    Source(slug="nhs-infertility", condition=CONDITIONS["infertility"],
           url="https://www.nhs.uk/conditions/infertility/"),
    Source(slug="nih-endometriosis", condition=CONDITIONS["endometriosis"],
           url="https://www.nichd.nih.gov/health/topics/endometri/conditioninfo"),
]


def snapshot_path(slug: str) -> Path:
    return RAW_DIR / f"{slug}.html"
