`social-preview.png` is rendered from `social-preview.html` at exactly
1200x630 (device scale 2):

    google-chrome --headless=new --disable-gpu --no-sandbox \
      --screenshot=docs/social-preview.png --window-size=1200,630 \
      --force-device-scale-factor=2 --hide-scrollbars \
      file://$PWD/docs/assets/social-preview.html

The three stats in the banner and the hand #341 board it shows are copied
from the README's tables; regenerate both whenever those numbers change.
