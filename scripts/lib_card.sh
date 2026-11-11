# THE CARD — the philosophy, handed to every agent at every launch.
#
# Owner, 2026-11-11: "every step must be aware of the philosophy it must nit
# forget". So it is not left to each prompt to remember: every launcher sources
# this and prepends `philosophy_card` to whatever it sends the model. One source
# (GOAL.md, between the CARD markers), six launch points, no copies to drift.
#
# UNKNOWN IS NOT ZERO, here too: if the card cannot be read, the agent is NOT
# launched philosophy-less. It gets the fallback below, which says so and asks
# for the repair — and `card_readable` lets the launcher log it loudly.
_card_raw() {
  sed -n '/<!-- CARD:BEGIN/,/<!-- CARD:END/p' "${REPO:-/home/opc/jackthelearner}/GOAL.md" 2>/dev/null | sed '1d;$d'
}
card_readable() { [ "$(_card_raw | wc -c)" -ge 600 ]; }
philosophy_card() {
  if card_readable; then
    _card_raw
  else
    cat <<'FALLBACK'
**THE PHILOSOPHY (FALLBACK — the card in GOAL.md could not be read; add one line
under INBOX in docs/ORDERS.md saying so, then carry on).** A creature with a
body, every human sense, one learning brain and real needs, in a world that can
kill him, who learns everything himself by living, dying and remembering.
Inside him nothing frozen or borrowed; pretrained models are teachers around
him. Every design choice is raced, never assumed. Nothing is claimed without a
test that could have failed; thresholds are never weakened; failures are fixed.
Free compute only; paying customers share this box.
FALLBACK
  fi
  printf '\n---\n\n'
}
