"""Hand-written, unambiguously public-domain (traditional / centuries-old)
ABC snippets used purely to exercise the tokenizer offline, without any
network access. These are NOT used as real training data for the shipped
models — they only back the automated tests in this sandbox, where the
actual genre data-source APIs (thesession.org / Hugging Face
datasets-server) are not reachable.
"""

TWINKLE_TWINKLE = """\
X:1
T:Twinkle Twinkle Little Star (fragment, traditional/public domain)
M:4/4
L:1/4
K:C
C C G G | A A G2 | F F E E | D D C2 |
"""

TIE_AND_KEY_SIGNATURE = """\
X:2
T:Tie And D-major Key Signature Test
M:4/4
L:1/8
K:D
F2 F2 F4 | G2-G2 A2 z2 |
"""

OCTAVE_AND_REST = """\
X:3
T:Octave Marks And Rest Test
M:3/4
L:1/4
K:C
C, E G | c e z |]
"""

HEADER_ONLY_NO_BODY = """\
X:4
T:Header Only, No Notes
M:4/4
L:1/8
K:C
"""

# A short, plausible "thesession.org"-style reel in ABC, used to test the
# data-source JSON field extraction logic without hitting the network.
SAMPLE_RESPONSE_ABC_REEL = """\
X:1
T:The Sandbox Reel
R:reel
M:4/4
L:1/8
K:Ador
A2BA GABd|eaag edBd|A2BA GABd|eaag edBA:|
"""
