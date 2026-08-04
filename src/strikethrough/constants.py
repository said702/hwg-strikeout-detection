LABEL_CLEAN = 0
LABEL_STRIKE = 1

LABEL_NAMES = {
    LABEL_CLEAN: "non-struck-out",
    LABEL_STRIKE: "struck-out",
}

NAME_TO_LABEL = {
    "clean": LABEL_CLEAN,
    "no": LABEL_CLEAN,
    "none": LABEL_CLEAN,
    "non-struck-out": LABEL_CLEAN,
    "no_strike": LABEL_CLEAN,
    "strike": LABEL_STRIKE,
    "struck-out": LABEL_STRIKE,
    "struck": LABEL_STRIKE,
}

TYPE_ALIASES = {
    "no": "none",
    "none": "none",
    "clean": "none",
    "sh": "single-horizontal",
    "single-horizontal": "single-horizontal",
    "single_horizontal": "single-horizontal",
    "so": "single-oblique",
    "single-oblique": "single-oblique",
    "single_oblique": "single-oblique",
    "mh": "multiple-horizontal",
    "multiple-horizontal": "multiple-horizontal",
    "multiple_horizontal": "multiple-horizontal",
    "mo": "multiple-oblique",
    "multiple-oblique": "multiple-oblique",
    "multiple_oblique": "multiple-oblique",
    "cr": "crossed",
    "crossed": "crossed",
    "ci": "circled",
    "circled": "circled",
    "wa": "wavy",
    "wavy": "wavy",
    "wave": "wavy",
    "zi": "zigzag",
    "zigzag": "zigzag",
    "zig_zag": "zigzag",
    "zig-zag": "zigzag",
    "bl": "blackened",
    "blackened": "blackened",
    "single_line": "single-horizontal",
    "double_line": "multiple-horizontal",
    "diagonal": "single-oblique",
    "cross": "crossed",
    "scratch": "blackened",
    "unknown": "unknown",
}

STRIKE_TYPES = [
    "single-horizontal",
    "single-oblique",
    "multiple-horizontal",
    "multiple-oblique",
    "crossed",
    "circled",
    "wavy",
    "zigzag",
    "blackened",
]

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
