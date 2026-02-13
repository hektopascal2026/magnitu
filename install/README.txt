Magnitu – Setup Instructions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Magnitu is a machine-learning relevance engine for Seismo.
It learns which entries matter to you and highlights investigation leads.

Requirements:
  • macOS or Linux
  • Python 3.9 or newer (https://www.python.org/downloads/)

Setup (one time):
  1. Open Terminal
  2. Navigate to this folder:  cd path/to/magnitu
  3. Run:  bash install/setup.sh
  4. Enter your API key when prompted
     (find it in Seismo → Settings → Magnitu section)

Daily use:
  • Double-click "Magnitu.command" on your Desktop
  • Or run:  cd path/to/magnitu && source .venv/bin/activate && python -m uvicorn main:app --port 8000
  • Open http://localhost:8000 in your browser

What it does:
  • Pulls entries from your Seismo instance
  • You label entries as: investigation_lead, important, background, noise
  • After ~20 labels, train a model
  • The model scores all entries and pushes results back to Seismo
  • Over time, it learns what matters to you
