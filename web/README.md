# PillTally web app

The phone page (Next.js). `app/page.tsx` uploads a photo to the model server (`src/api.py`, port 8000)
and draws the boxes; `app/zoom-pan.tsx` handles pinch-zoom.
Install with `npm ci`; start it with `npx next dev -H 0.0.0.0 -p 3000`, with the model server running (see the repo README).
