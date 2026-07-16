import re

from bs4 import BeautifulSoup
from pydantic import BaseModel

BOILERPLATE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]
MIN_SECTION_CHARS = 40
_JUNK_CLASS_RE = re.compile(r"breadcrumb|share|social", re.IGNORECASE)


class Section(BaseModel):
    heading: str
    text: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_sections(html: str) -> list[Section]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(BOILERPLATE_TAGS):
        tag.decompose()
    for tag in soup.find_all(attrs={"class": _JUNK_CLASS_RE}):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": "navigation"}):
        tag.decompose()

    root = soup.find("main") or soup.body or soup
    sections: list[Section] = []
    current_heading: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_heading, current_parts
        if current_heading:
            text = _clean(" ".join(current_parts))
            if len(text) >= MIN_SECTION_CHARS:
                sections.append(Section(heading=current_heading, text=text))
        current_parts = []

    for el in root.find_all(["h1", "h2", "h3", "p", "li"]):
        if el.name in ("h1", "h2", "h3"):
            flush()
            current_heading = _clean(el.get_text())
        else:
            if el.name == "p" and el.find_parent("li") is not None:
                continue
            current_parts.append(el.get_text())
    flush()
    return sections
