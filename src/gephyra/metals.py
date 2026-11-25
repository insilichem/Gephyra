"""
Metal recognition and metal-water coordination scoring.

A metal can be the starting point (root) of both techniques. Users name it as
their topology does - `Cu`, `CU`, `CU1`, `Zn2+`, `ZN` - and AMBER/MCPB-style
topologies often leave the element of a parameterised metal empty, so a metal
is recognised from, in order: the element attribute, the atom name, the
residue name (single-atom residues), and finally the mass.

The first link of a path that starts at a metal is not a hydrogen bond but a
coordination bond, so it is scored on the metal-oxygen distance alone (no
hydrogen geometry), around the ideal distance for that metal.

Ideal M-O(water) distances are approximate means for the common oxidation
state from surveys of crystal structures (M. M. Harding, Acta Cryst. D 57,
401-411 (2001); D 62, 678-682 (2006)) and standard ionic radii where no survey
value exists. They set where the coordination score starts to decay; they are
not a force field.
"""

import re

# symbol -> (name, atomic mass, ideal metal-to-water-oxygen distance in A)
METALS = {
    "LI": ("lithium", 6.94, 1.96), "NA": ("sodium", 22.99, 2.41), "K": ("potassium", 39.10, 2.81),
    "RB": ("rubidium", 85.47, 2.95), "CS": ("caesium", 132.91, 3.10), "BE": ("beryllium", 9.01, 1.62),
    "MG": ("magnesium", 24.31, 2.07), "CA": ("calcium", 40.08, 2.39), "SR": ("strontium", 87.62, 2.58),
    "BA": ("barium", 137.33, 2.79), "AL": ("aluminium", 26.98, 1.89), "TI": ("titanium", 47.87, 2.02),
    "V": ("vanadium", 50.94, 2.00), "CR": ("chromium", 52.00, 2.00), "MN": ("manganese", 54.94, 2.19),
    "FE": ("iron", 55.85, 2.10), "CO": ("cobalt", 58.93, 2.09), "NI": ("nickel", 58.69, 2.07),
    "CU": ("copper", 63.55, 2.00), "ZN": ("zinc", 65.38, 2.09), "GA": ("gallium", 69.72, 1.95),
    "MO": ("molybdenum", 95.95, 2.10), "RU": ("ruthenium", 101.07, 2.10), "RH": ("rhodium", 102.91, 2.05),
    "PD": ("palladium", 106.42, 2.05), "AG": ("silver", 107.87, 2.40), "CD": ("cadmium", 112.41, 2.31),
    "SN": ("tin", 118.71, 2.20), "W": ("tungsten", 183.84, 2.10), "IR": ("iridium", 192.22, 2.05),
    "PT": ("platinum", 195.08, 2.05), "AU": ("gold", 196.97, 2.10), "HG": ("mercury", 200.59, 2.38),
    "PB": ("lead", 207.2, 2.50), "LA": ("lanthanum", 138.91, 2.55), "GD": ("gadolinium", 157.25, 2.42),
    "EU": ("europium", 151.96, 2.45), "TB": ("terbium", 158.93, 2.40), "YB": ("ytterbium", 173.05, 2.32),
}
MASS_TOLERANCE = 1.0                  # amu, for recognising a metal by mass
COORDINATION_WIDTH = 0.6              # A beyond the ideal distance at which the score reaches 0
COORDINATION_STERIC = 0.35            # A below the ideal distance at which a clash penalty starts

_SYMBOL_RE = re.compile(r"^([A-Za-z]{1,2})(?:[0-9]*[+-]*[0-9]*|[+-][0-9]*)$")


def _symbol_from_text(text):
    """'Cu', 'CU', 'CU1', 'Zn2+', 'ZN+2', 'Na+' -> metal symbol, else None."""
    text = str(text or "").strip()
    match = _SYMBOL_RE.match(text)
    if not match:
        return None
    letters = match.group(1).upper()
    return letters if letters in METALS else None


_POLYMER_RESNAMES = None


def _is_polymer_residue(atom):
    """Protein or nucleic-acid residue, by MDAnalysis' residue-name lists."""
    global _POLYMER_RESNAMES
    if _POLYMER_RESNAMES is None:
        try:
            from MDAnalysis.core.selection import ProteinSelection, NucleicSelection
            _POLYMER_RESNAMES = set(ProteinSelection.prot_res) | set(NucleicSelection.nucl_res)
        except Exception:  # pragma: no cover - very old MDAnalysis
            _POLYMER_RESNAMES = set()
    return atom.resname.strip().upper() in _POLYMER_RESNAMES


def metal_symbol(atom):
    """
    The metal's element symbol (e.g. 'CU') if `atom` is a metal, else None.

    Order: element attribute; then, outside protein and nucleic-acid residues
    (so that a C-alpha named 'CA' is never taken for calcium), the atom name,
    the residue name of a single-atom residue, and the mass.
    """
    element = getattr(atom, "element", "") or ""
    element = str(element).strip().upper()
    if element:
        return element if element in METALS else None
    if _is_polymer_residue(atom):
        return None
    for text in (atom.name, atom.resname if len(atom.residue.atoms) == 1 else None):
        symbol = _symbol_from_text(text)
        if symbol:
            mass = _mass(atom)
            # A name like 'CA' or 'NA' could still be a light atom: trust the
            # mass when the topology provides one.
            if mass is None or mass < 2 or abs(mass - METALS[symbol][1]) <= 3 * MASS_TOLERANCE:
                return symbol
    mass = _mass(atom)
    if mass is not None and len(atom.residue.atoms) == 1:
        for symbol, (_, metal_mass, _) in METALS.items():
            if abs(mass - metal_mass) <= MASS_TOLERANCE:
                return symbol
    return None


def _mass(atom):
    try:
        mass = float(atom.mass)
    except Exception:
        return None
    return mass if mass > 0 else None


def describe(symbol):
    name, _, distance = METALS[symbol]
    return f"{symbol.capitalize()} ({name}, ideal {symbol.capitalize()}-O {distance:.2f} A)"


def metal_atoms(atomgroup):
    """[(atom, symbol)] for the metals in an AtomGroup."""
    return [(a, s) for a in atomgroup for s in [metal_symbol(a)] if s]


def coordination_probability(distance, symbol):
    """
    Score in [0, 1] for a metal-oxygen coordination bond of length `distance`:
    1 up to the ideal distance, decaying smoothly to 0 at ideal + 0.6 A, and
    penalised for clashes shorter than ideal - 0.35 A.
    """
    from .math_utils import switching_function
    ideal = METALS[symbol][2]
    p = switching_function(max(distance - ideal, 0.0), threshold=COORDINATION_WIDTH)
    clash = ideal - COORDINATION_STERIC
    if distance < clash:
        p *= (distance / clash) ** 12
    return float(p)
