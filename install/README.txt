Magnitu – Installation
━━━━━━━━━━━━━━━━━━━━━━

Magnitu is a machine-learning relevance engine for Seismo.
It learns which news entries matter to you and highlights investigation leads.

Requirements:
  • macOS (or Linux)
  • Python 3.9+ (pre-installed on macOS)

Install (one-liner):
  Open Terminal and paste:

  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/hektopascal2026/magnitu/main/install/bootstrap.sh)"

  It will:
  1. Download Magnitu to ~/magnitu
  2. Set up the Python environment
  3. Ask for your API key
  4. Test the connection to Seismo

Start Magnitu:
  ~/magnitu/start.sh

  It opens automatically in your browser at http://localhost:8000

Update:
  cd ~/magnitu && git pull

Uninstall:
  rm -rf ~/magnitu
