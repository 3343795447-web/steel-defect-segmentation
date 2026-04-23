import re
import sys
import zipfile


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: py read_docx_text.py <path-to-docx>")
    path = sys.argv[1]
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", "ignore")
    text = re.sub(r"<[^>]+>", "\n", xml)
    text = re.sub(r"\n+", "\n", text)
    print(text[:50000])


if __name__ == "__main__":
    main()

