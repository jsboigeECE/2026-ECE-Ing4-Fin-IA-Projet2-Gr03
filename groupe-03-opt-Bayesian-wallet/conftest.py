import sys
import os

# Ajoute le dossier src/ au path pour que pytest trouve les modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
